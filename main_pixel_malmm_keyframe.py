#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_pixel_malmm_keyframe.py
Pixel-space PGD attack for MA-LMM (LAVIS BLIP2 Vicuna Instruct MA-LMM)
- Keep MA-LMM fps=10 frame sampling alignment (default: interval-based sampling)
- Optional timestamp-based sampling for stricter time alignment
- Keyframe-cluster shared perturbation (Plan B: Representative-driven Shared Perturbation)
- DDP data parallel over videos
"""

import os
import sys
import gc
import csv
import json
import argparse
import traceback
from typing import Dict, List, Any, Optional, Tuple
import subprocess

import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
from torch.cuda.amp import autocast

from torchvision import transforms
from transformers import LlamaTokenizer
from lavis.models import load_model_and_preprocess


# -------------------------
# DDP
# -------------------------
def init_dist() -> Tuple[int, int, int, torch.device, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    is_master = (rank == 0)
    return local_rank, rank, world_size, device, is_master


# -------------------------
# Frame sampling
# -------------------------
def sample_frames_interval(video_path: str, fps: int = 10, max_frames: Optional[int] = None) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    v_fps = cap.get(cv2.CAP_PROP_FPS)
    if v_fps <= 0:
        v_fps = 25.0

    frame_interval = max(1, int(v_fps / float(fps)))

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        count += 1

    cap.release()
    frames = np.array(frames)
    if frames.size == 0:
        frames = sample_frames_ffmpeg(video_path, fps=fps, max_frames=max_frames)
    return frames

def sample_frames_ffmpeg(video_path: str, fps: int = 10, max_frames: Optional[int] = None) -> np.ndarray:
    try:
        wh = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            video_path
        ]).decode().strip()
        w, h = map(int, wh.split("x"))
    except Exception as e:
        print(f"[FFPROBE] failed on {video_path}: {e}", flush=True)
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)

    vframes = max_frames if max_frames is not None else 10**9

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-vframes", str(vframes),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    try:
        out = subprocess.check_output(cmd)
    except Exception as e:
        print(f"[FFMPEG] failed on {video_path}: {e}", flush=True)
        return np.zeros((0, h, w, 3), dtype=np.uint8)

    frame_size = w * h * 3
    n = len(out) // frame_size
    if n == 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8)

    arr = np.frombuffer(out[: n * frame_size], dtype=np.uint8)
    return arr.reshape(n, h, w, 3)

def sample_frames_timestamp(video_path: str, fps: int = 10, max_frames: Optional[int] = None) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    v_fps = cap.get(cv2.CAP_PROP_FPS)
    if v_fps <= 0:
        v_fps = 25.0

    duration_sec = frame_count / v_fps if frame_count and frame_count > 0 else None

    frames = []
    n = 0
    while True:
        t_sec = n / float(fps)
        if duration_sec is not None and t_sec > duration_sec:
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        n += 1
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    frames = np.array(frames)
    if frames.size == 0:
        frames = sample_frames_ffmpeg(video_path, fps=fps, max_frames=max_frames)
    return frames


# -------------------------
# QA loading
# -------------------------
def load_qa_from_csv(csv_path: str) -> Dict[str, List[Dict[str, str]]]:
    vid2qa: Dict[str, List[Dict[str, str]]] = {}
    if not os.path.exists(csv_path):
        return vid2qa

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id") or row.get("video")
            if not vid:
                continue
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not q or not a:
                continue

            vid_key = vid.replace("video", "").split(".")[0]
            vid2qa.setdefault(vid_key, []).append({"question": q, "answer": a})
    return vid2qa


class VideoListDataset(Dataset):
    def __init__(self, folder_path: str):
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mp4")])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> str:
        return self.files[idx]


# -------------------------
# Keyframe clustering (cosine k-means on frame embeddings)
# -------------------------
def cosine_kmeans(frame_vecs: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    T, D = frame_vecs.shape
    if T <= k:
        return torch.arange(T, device=frame_vecs.device, dtype=torch.long)

    idx = torch.randperm(T, device=frame_vecs.device)[:k]
    centroids = frame_vecs[idx]
    labels = None

    for _ in range(iters):
        sim = frame_vecs @ centroids.t()
        labels = sim.argmax(dim=1)

        new_centroids = []
        for ci in range(k):
            mask = (labels == ci)
            if mask.any():
                c = frame_vecs[mask].mean(dim=0)
                c = F.normalize(c, dim=0)
            else:
                c = frame_vecs[torch.randint(0, T, (1,), device=frame_vecs.device)].squeeze(0)
            new_centroids.append(c)
        centroids = torch.stack(new_centroids, dim=0)

    return labels.long()


@torch.no_grad()
def build_frame_mapping_by_vit(
    original_images_fp32: torch.Tensor,
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use MA-LMM visual encoder embeddings to cluster frames and find representative keyframes.
    Returns: mapping (frame to cluster), rep_idx (cluster to representative frame index)
    """
    norm = (original_images_fp32 - mean) / std
    norm_fp16 = norm.to(device=device, dtype=torch.float16)

    with autocast(enabled=True, dtype=torch.float16):
        vit_out = model.visual_encoder(norm_fp16)
        image_embeds = model.ln_vision(vit_out)

    frame_vec = image_embeds.mean(dim=1).float()
    frame_vec = F.normalize(frame_vec, dim=1)
    mapping = cosine_kmeans(frame_vec, k=k, iters=10)

    # 选出每个簇的中心关键帧 (Plan B 核心修改)
    actual_k = int(mapping.max().item() + 1)
    rep_idx = torch.zeros(actual_k, dtype=torch.long, device=device)
    for ci in range(actual_k):
        mask = (mapping == ci)
        if not mask.any():
            continue
        cluster_vecs = frame_vec[mask]
        centroid = F.normalize(cluster_vecs.mean(dim=0, keepdim=True), dim=1)
        sims = (cluster_vecs @ centroid.t()).squeeze(1)
        best_local_idx = sims.argmax()
        global_indices = torch.nonzero(mask).squeeze(1)
        rep_idx[ci] = global_indices[best_local_idx]

    return mapping, rep_idx


# -------------------------
# Pixel PGD attack (Plan B: Representative-driven)
# -------------------------
def run_pixel_attack_on_malmm(
    frames_uint8_thwc: torch.Tensor,
    qa_list: List[Dict[str, str]],
    model,
    tokenizer: LlamaTokenizer,
    device: torch.device,
    eps: float,
    alpha: float,
    steps: int,
    num_keyframes: int,
    qmax_words: int,
    qa_batch_size: int,
    random_start: bool = False,
    save_adv_frames: bool = False,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:

    # 1. 模型设为 Eval
    model.eval()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad = False

    # 2. 图像预处理参数
    image_size = 224
    prep_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    # 3. 原始图像处理 [T, 3, 224, 224]
    processed = [prep_transform(Image.fromarray(fr.numpy())) for fr in frames_uint8_thwc]
    original_images = torch.stack(processed).to(device=device, dtype=torch.float32)
    T = original_images.shape[0]

    # 4. 初始化扰动 Delta
    if num_keyframes and num_keyframes > 0:
        k = max(1, min(int(num_keyframes), T))
        mapping, rep_idx = build_frame_mapping_by_vit(original_images, model, mean, std, device, k)
        actual_k = int(mapping.max().item() + 1)
        delta_key = torch.zeros((actual_k, 3, image_size, image_size), device=device, dtype=torch.float32, requires_grad=True)
        if random_start:
            delta_key.data.uniform_(-eps, eps)
        use_key = True
    else:
        mapping = None
        rep_idx = None
        delta = torch.zeros_like(original_images, device=device, dtype=torch.float32, requires_grad=True)
        if random_start:
            delta.data.uniform_(-eps, eps)
        use_key = False

    # 5. Tokenizer 处理
    questions = [x["question"] for x in qa_list]
    answers = [x["answer"] for x in qa_list]
    prompts = [f"Question: {q} Answer:" for q in questions]
    full_texts = [f"Question: {q} Answer: {a}{tokenizer.eos_token}" for q, a in zip(questions, answers)]

    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=qmax_words).to(device)
    prompt_lens = prompt_tokens.attention_mask.sum(dim=1).long()
    full_tokens = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    Q = len(questions)

    if hasattr(model.llm_model, "gradient_checkpointing_enable"):
        model.llm_model.gradient_checkpointing_enable()

    try:
        # ==================== PGD 攻击循环 ====================
        for step_idx in range(steps):

            # 计算 adv_images (Plan B 核心修改)
            if use_key:
                if delta_key.grad is not None: delta_key.grad.zero_()

                delta_full_list = []
                for t in range(T):
                    c = mapping[t]
                    if t == rep_idx[c]:
                        # 关键帧：保留梯度
                        delta_full_list.append(delta_key[c])
                    else:
                        # 非关键帧：切断梯度，仅复用扰动值
                        delta_full_list.append(delta_key[c].detach())

                delta_full = torch.stack(delta_full_list, dim=0)
                adv_images = torch.clamp(original_images + delta_full, 0.0, 1.0)
            else:
                if delta.grad is not None: delta.grad.zero_()
                adv_images = torch.clamp(original_images + delta, 0.0, 1.0)

            norm_images = (adv_images - mean) / std
            norm_images_fp16 = norm_images.to(torch.float16)

            # 前向传播计算梯度
            with autocast(enabled=True, dtype=torch.float16):
                vit_out = model.visual_encoder(norm_images_fp16)
                image_embeds = model.ln_vision(vit_out)

                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
                query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = model.Qformer.bert(query_embeds=query_tokens, encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True)
                video_features = model.llm_proj(query_output.last_hidden_state)
                video_feature_agg = video_features.mean(dim=0, keepdim=True)

                total_loss_val = 0.0
                for i in range(0, Q, qa_batch_size):
                    bs = min(qa_batch_size, Q - i)
                    batch_video_input = video_feature_agg.expand(bs, -1, -1)
                    full_input_ids = full_tokens.input_ids[i:i+bs]
                    full_attn = full_tokens.attention_mask[i:i+bs]
                    batch_prompt_lens = prompt_lens[i:i+bs]

                    full_text_embeds = model.llm_model.get_input_embeddings()(full_input_ids)
                    inputs_embeds = torch.cat([batch_video_input, full_text_embeds], dim=1)
                    attention_mask = torch.cat([torch.ones(batch_video_input.shape[:2], dtype=torch.long, device=device), full_attn], dim=1)

                    labels_text = full_input_ids.clone()
                    for r in range(bs):
                        pl = int(batch_prompt_lens[r].item())
                        labels_text[r, :pl] = -100
                    labels_text[labels_text == tokenizer.pad_token_id] = -100
                    labels = torch.cat([torch.full((bs, batch_video_input.shape[1]), -100, dtype=torch.long, device=device), labels_text], dim=1)

                    outputs = model.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, return_dict=True)
                    loss = outputs.loss * (bs / float(Q))

                    retain_graph = (i + qa_batch_size < Q)
                    loss.backward(retain_graph=retain_graph)

            # 更新 delta
            with torch.no_grad():
                if use_key:
                    grad_sign = delta_key.grad.sign()
                    delta_key.data = delta_key.data + alpha * grad_sign
                    delta_key.data = torch.clamp(delta_key.data, -eps, eps)
                else:
                    grad_sign = delta.grad.sign()
                    delta.data = delta.data + alpha * grad_sign
                    delta.data = torch.clamp(delta.data, -eps, eps)
        # ==================== 攻击结束 ====================

        # ==================== 保存特征 ====================
        with torch.no_grad(), autocast(enabled=True, dtype=torch.float16):
            # 1. 生成最终对抗图像 (为防止显存泄露，附加 detach)
            if use_key:
                final_adv = torch.clamp(original_images + delta_key[mapping].detach(), 0.0, 1.0)
            else:
                final_adv = torch.clamp(original_images + delta.detach(), 0.0, 1.0)

            # 2. 归一化
            final_norm = (final_adv - mean) / std

            # 3. 仅通过 Visual Encoder 提取 Raw Features
            adv_raw_feats = model.visual_encoder(final_norm.to(torch.float16))

            # 可视化需求
            adv_frames_np = None
            if save_adv_frames:
                final_adv_cpu = final_adv.detach().cpu().clamp(0, 1)
                adv_frames_np = (final_adv_cpu.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)

        return adv_raw_feats.detach().cpu(), adv_frames_np

    finally:
        if hasattr(model.llm_model, "gradient_checkpointing_disable"):
            model.llm_model.gradient_checkpointing_disable()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--base_dir", type=str, default="./data/msrvtt/")
    p.add_argument("--videos_dir", type=str, default="", help="Override videos dir")
    p.add_argument("--qa_csv", type=str, default="./just-ask/MSRVTT-QA/test.csv")
    p.add_argument("--log_dir", type=str, default="./outputs")
    p.add_argument("--out_dir", type=str, default="", help="Override out dir")

    p.add_argument("--vicuna_path", type=str, default="./checkpoints/MA-LMM/llm/vicuna-7b/")
    p.add_argument("--eva_ckpt", type=str, default="./checkpoints/MA-LMM/pretrained/eva_vit_g.pth")
    p.add_argument("--malmm_ckpt", type=str, default="./checkpoints/MA-LMM/pretrained/instruct_blip_vicuna7b_trimmed.pth")

    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max_frames", type=int, default=20)
    p.add_argument("--timestamp_sampling", "--timestamp-sampling", action="store_true")

    p.add_argument("--eps", type=float, default=8.0/255.0)
    p.add_argument("--alpha", type=float, default=2.0/255.0)
    p.add_argument("--steps", type=int, default=10)

    p.add_argument("--num_keyframes", type=int, default=8)
    p.add_argument("--random_start", "--random-start", action="store_true")

    p.add_argument("--qmax_words", type=int, default=32)
    p.add_argument("--qa_batch_size", type=int, default=4)

    p.add_argument("--save_adv_frames", "--save-adv-frames", action="store_true")
    p.add_argument("--adv_frames_dir", type=str, default="")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--local_rank", "--local-rank", type=int, default=0, help="Ignored")

    args, _ = p.parse_known_args()
    return args


def save_adv_frames_as_jpg(adv_frames_uint8_thwc: np.ndarray, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for i, fr in enumerate(adv_frames_uint8_thwc):
        Image.fromarray(fr).save(os.path.join(save_dir, f"frame_{i:06d}.jpg"), quality=95)


def main():
    args = parse_args()
    local_rank, rank, world_size, device, is_master = init_dist()

    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)

    videos_dir = args.videos_dir if args.videos_dir else os.path.join(args.base_dir, "videos")
    out_dir = args.out_dir if args.out_dir else os.path.join(args.log_dir, "data/feats_pgd_answer_keyframe")
    os.makedirs(out_dir, exist_ok=True)

    adv_frames_root = args.adv_frames_dir if args.adv_frames_dir else os.path.join(out_dir, "adv_frames")

    if is_master:
        print(f"[Rank {rank}] device={device} world_size={world_size}", flush=True)
        print(f"videos_dir: {videos_dir}", flush=True)
        print(f"out_dir: {out_dir}", flush=True)
        print(f"fps={args.fps} max_frames={args.max_frames} sampling={'timestamp' if args.timestamp_sampling else'interval'}", flush=True)
        print(f"PGD eps={args.eps} alpha={args.alpha} steps={args.steps} num_keyframes={args.num_keyframes} random_start={args.random_start}", flush=True)

    if is_master:
        print("[Model] Loading MA-LMM (lavis)...", flush=True)

    model, _, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct_malmm",
        model_type="vicuna7b",
        is_eval=True,
        device=device,
    )

    tokenizer = LlamaTokenizer.from_pretrained(args.vicuna_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.llm_tokenizer = tokenizer

    eva_ckpt = torch.load(args.eva_ckpt, map_location="cpu")
    model.visual_encoder.load_state_dict(eva_ckpt.get("model", eva_ckpt), strict=False)

    malmm_ckpt = torch.load(args.malmm_ckpt, map_location="cpu")
    model.load_state_dict(malmm_ckpt, strict=False)

    model = model.to(device=device, dtype=torch.float16)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if is_master:
        print("[Model] Ready.", flush=True)

    vid2qa = load_qa_from_csv(args.qa_csv)
    ds = VideoListDataset(videos_dir)

    remaining_indices = []
    if rank == 0:
        for idx, fname in enumerate(ds.files):
            vid_key = fname.split("video")[-1].split(".")[0]
            if vid_key not in vid2qa:
                continue
            out_path = os.path.join(out_dir, f"{fname.split('.')[0]}_adv_feats.pt")
            if os.path.exists(out_path):
                continue
            remaining_indices.append(idx)
        print(f"[Rank 0] videos to process: {len(remaining_indices)}", flush=True)

    obj_list = [remaining_indices]
    dist.broadcast_object_list(obj_list, src=0)
    remaining_indices = obj_list[0]
    if not remaining_indices:
        dist.barrier()
        if is_master:
            print("[DONE] Nothing to do.", flush=True)
        dist.destroy_process_group()
        return

    subset = Subset(ds, remaining_indices)
    sampler = DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(subset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=lambda x: x)

    log_path = os.path.join(args.log_dir, f"log_pixel_attack.rank{rank}.txt")

    for batch in tqdm(loader, desc=f"[Rank {rank}]"):
        fname = batch[0]
        vid_key = fname.split("video")[-1].split(".")[0]
        video_path = os.path.join(videos_dir, fname)
        out_path = os.path.join(out_dir, f"{fname.split('.')[0]}_adv_feats.pt")

        try:
            if args.timestamp_sampling:
                frames_np = sample_frames_timestamp(video_path, fps=args.fps, max_frames=args.max_frames)
            else:
                frames_np = sample_frames_interval(video_path, fps=args.fps, max_frames=args.max_frames)

            if frames_np is None or len(frames_np) == 0:
                continue

            qa = vid2qa.get(vid_key, [])
            if not qa:
                continue

            frames_thwc = torch.from_numpy(np.stack(frames_np))

            adv_feats, adv_frames = run_pixel_attack_on_malmm(
                frames_uint8_thwc=frames_thwc,
                qa_list=qa,
                model=model,
                tokenizer=tokenizer,
                device=device,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                num_keyframes=args.num_keyframes,
                qmax_words=args.qmax_words,
                qa_batch_size=args.qa_batch_size,
                random_start=args.random_start,
                save_adv_frames=args.save_adv_frames,
            )

            torch.save(adv_feats, out_path)

            if args.save_adv_frames and adv_frames is not None:
                save_dir = os.path.join(adv_frames_root, fname.split(".")[0])
                save_adv_frames_as_jpg(adv_frames, save_dir)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"SUCCESS: {fname}\n")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            if is_master:
                print(f"[ERROR Rank {rank}] {fname}: {e}", flush=True)
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    dist.barrier()
    if is_master:
        print("[DONE] All finished.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
