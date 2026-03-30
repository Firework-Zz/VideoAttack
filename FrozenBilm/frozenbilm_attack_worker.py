#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrozenBiLM 像素级对抗攻击 Worker 脚本 (Pixel Attack)
Fix: 解决多卡并行时的 CUDA Device Ordinal 错误
"""

import os
import sys
import torch
import torch.nn.functional as F
import argparse
import json
import numpy as np
import cv2
import clip

# [配置] 强行指定 FrozenBiLM 根目录
FROZEN_ROOT = "/data2/codefile/fjh/FrozenBilm"
if FROZEN_ROOT not in sys.path: sys.path.insert(0, FROZEN_ROOT)

# [Patch] 修复 NumPy 兼容性
if not hasattr(np, 'int'): np.int = int

# 引入 FrozenBiLM 模块
try:
    from model import build_model, get_tokenizer
    from args import get_args_parser
    from util.misc import get_mask
except ImportError:
    print(f"[Worker] Error importing FrozenBiLM. Path: {sys.path}")
    raise

def load_video_frames(video_path, num_frames=10, size=224):
    cap = cv2.VideoCapture(video_path)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    if vlen <= 0:
        for _ in range(num_frames):
            frames.append(np.zeros((size, size, 3), dtype=np.uint8))
    else:
        indices = np.linspace(0, vlen - 1, num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((size, size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size, size))
            frames.append(frame)
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if len(frames) > 0 else np.zeros((size, size, 3), dtype=np.uint8))
    
    np_frames = np.stack(frames)
    tensor = torch.from_numpy(np_frames).float() 
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor.unsqueeze(0) / 255.0

def run_worker(args):
    # =========================================================================
    # 【关键修复】 强制使用 cuda:0
    # 原因：主脚本通过 CUDA_VISIBLE_DEVICES=x 启动子进程。
    # 在子进程内部，可见的显卡永远会被重映射为 cuda:0。
    # args.gpu_id 仅用于打印日志区分身份，不能用于 torch.device()
    # =========================================================================
    device = torch.device("cuda:0") 
    
    print(f"[GPU {args.gpu_id}] Initializing Pixel Attack Worker (Internal Device: cuda:0)...")

    # A. 加载 CLIP
    print(f"[GPU {args.gpu_id}] Loading CLIP...")
    clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters(): p.requires_grad = False
    
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1).to(device)

    # B. 加载 FrozenBiLM
    print(f"[GPU {args.gpu_id}] Loading FrozenBiLM...")
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    
    dummy_args = [
        "--combine_datasets", "msvd",
        "--combine_datasets_val", "msvd",
        "--model_name", "deberta-v2-xlarge", 
        "--device", str(device)
    ]
    fb_args = parser.parse_known_args(dummy_args)[0]
    
    # 路径覆盖
    fb_args.model_name = "/data0/pretrained/deberta-v2-xlarge/"
    
    if args.checkpoint:
        fb_args.load = args.checkpoint
    else:
        fb_args.load = "/data0/pretrained/FrozenBilm/checkpoints/frozenbilm_msvd.pth"
        
    fb_args.max_tokens = 256
    
    tokenizer = get_tokenizer(fb_args)
    model = build_model(fb_args)
    model.to(device)
    
    print(f"[GPU {args.gpu_id}] Loading Checkpoint: {fb_args.load}")
    try:
        checkpoint = torch.load(fb_args.load, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return

    model.eval()
    for param in model.parameters(): param.requires_grad = False

    # 加载任务
    with open(args.task_file, 'r') as f:
        tasks = json.load(f)
    print(f"[GPU {args.gpu_id}] Received {len(tasks)} videos.")

    # 攻击循环
    count = 0
    for task in tasks:
        video_path = task['video_path']
        video_id = str(task['video_id'])
        qa_pairs = task['qa_pairs']
        
        save_path = os.path.join(args.output_dir, f"{video_id}.pth")
        if os.path.exists(save_path): continue

        try:
            raw_frames = load_video_frames(video_path, num_frames=10).to(device)
            
            delta = torch.zeros_like(raw_frames).to(device)
            delta.uniform_(-args.eps, args.eps)
            delta.requires_grad = True
            
            for step in range(args.steps):
                adv_frames = torch.clamp(raw_frames + delta, 0, 1)
                
                adv_norm = (adv_frames - mean) / std
                b, t, c, h, w = adv_norm.shape
                adv_input = adv_norm.view(-1, c, h, w)
                
                adv_features = clip_model.encode_image(adv_input).float()
                adv_features = adv_features.view(b, t, -1)
                
                total_loss = 0
                valid_qa_count = 0
                
                for qa in qa_pairs:
                    q_text = qa['q']
                    a_text = qa['a']
                    full_text = q_text + " Answer: [MASK]."
                    
                    encoded_q = tokenizer([full_text], return_tensors="pt", padding="longest", truncation=True, max_length=fb_args.max_tokens)
                    input_ids = encoded_q.input_ids.to(device)
                    attention_mask = encoded_q.attention_mask.to(device)
                    
                    encoded_a = tokenizer([a_text], add_special_tokens=False, return_tensors="pt")
                    if encoded_a.input_ids.size(1) > 0:
                        answer_id = encoded_a.input_ids[0, 0].to(device)
                    else: continue

                    output = model(
                        video=adv_features,
                        video_mask=get_mask(torch.tensor([t]), t).to(device),
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    logits = output["logits"]
                    mask_pos = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
                    
                    if len(mask_pos[0]) > 0:
                        offset = t
                        target_logits = logits[mask_pos[0], mask_pos[1] + offset, :]
                        loss = F.cross_entropy(target_logits, answer_id.view(-1))
                        total_loss += loss
                        valid_qa_count += 1
                
                if valid_qa_count == 0: break

                total_loss = total_loss / valid_qa_count

                model.zero_grad()
                clip_model.zero_grad()
                total_loss.backward()
                
                grad = delta.grad.detach()
                delta.data = delta.data + args.alpha * torch.sign(grad)
                delta.data = torch.clamp(delta.data, -args.eps, args.eps)
                delta.grad.zero_()

            with torch.no_grad():
                final_frames = torch.clamp(raw_frames + delta, 0, 1)
                final_norm = (final_frames - mean) / std
                final_input = final_norm.view(-1, c, h, w)
                final_feat = clip_model.encode_image(final_input).float()
                final_feat = final_feat.view(t, -1).cpu()
                torch.save(final_feat, save_path)
                count += 1
                
        except Exception as e:
            # print(f"[GPU {args.gpu_id}] Error {video_id}: {e}")
            continue

    print(f"[GPU {args.gpu_id}] Finished. Processed {count} videos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--eps", type=float, default=8.0/255.0)
    parser.add_argument("--alpha", type=float, default=2.0/255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()
    
    run_worker(args)
