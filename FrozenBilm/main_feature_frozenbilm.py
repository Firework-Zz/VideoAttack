#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import re
import json
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# [NumPy Patch]
# -----------------------------------------------------------------------------
if not hasattr(np, 'int'):
    np.int = int

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# -----------------------------------------------------------------------------
# 1. Configuration & Paths (MSRVTT Setup)
# -----------------------------------------------------------------------------

# FrozenBiLM 项目路径
FROZEN_ROOT = "/data2/codefile/fjh/FrozenBilm"
if FROZEN_ROOT not in sys.path: sys.path.insert(0, FROZEN_ROOT)

try:
    from model import build_model, get_tokenizer
    from util.misc import get_mask
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    sys.exit(1)

# --- 输入路径 (MSRVTT) ---
# [请确认] MSRVTT 的 CLIP 特征文件路径
CLEAN_FEATS_PATH = "/data0/data/MSRVTT-QA/clipvitl14.pth"
# [请确认] MSRVTT-QA 的 CSV 路径
ANS_CSV = "/data0/data/msrvtt/MSRVTT-QA/test.csv"

# --- 模型权重 ---
DEBERTA_PATH = "/data0/pretrained/deberta-v2-xlarge/" 
# MSRVTT 权重
CHECKPOINT_PATH = "/data0/pretrained/FrozenBilm/checkpoints/frozenbilm_msrvtt.pth"

# --- 输出路径 ---
RESULTS_BASE_DIR = "/data2/codefile/fjh/FrozenBilm/results_msrvtt"
OUT_DIR = os.path.join(RESULTS_BASE_DIR, "feats_frozenbilm_10")
FEATS_SAVE_DIR = os.path.join(OUT_DIR, "adv_features_npy")

# --- 攻击参数 ---
FEAT_EPS = 0.5      
FEAT_ALPHA = 0.1  
FEAT_STEPS = 10       
RESTARTS = 1

# [显存控制] QA 分批大小
QA_BATCH_SIZE = 32  

# 视频ID过滤范围
RANGE_LOW, RANGE_HIGH = 0, 99999
MAX_TOKENS = 256 

# -----------------------------------------------------------------------------
# 2. Utils & DDP Setup
# -----------------------------------------------------------------------------

def init_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=60))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        dist.init_process_group(backend="gloo", init_method="env://")
        device = torch.device("cpu")
    
    is_master = (rank == 0)
    return local_rank, rank, world_size, device, is_master

def accuracy(logits, target_token_id, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target_token_id.size(0)
        
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target_token_id.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item() / batch_size)
        return res

def load_qa_from_csv(csv_path: str):
    """
    加载 QA 数据，适配 MSRVTT 格式
    通常包含 video_id, question, answer
    """
    vid2qa = {}
    if not os.path.exists(csv_path): return vid2qa
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            # 兼容不同列名
            vid = row.get("video_id") or row.get("video") or row.get("id")
            q = row.get("question")
            a = row.get("answer")
            if not vid or not q or not a: continue
            
            # 去除后缀和空格
            base = re.sub(r".mp4$", "", vid, flags=re.IGNORECASE).strip()
            vid2qa.setdefault(base, []).append((q, a))
            
            # MSRVTT 有时是 video0, video1... 尝试提取纯数字以备用
            num = re.sub(r"\D", "", base)
            if num and num != base:
                vid2qa.setdefault(num, []).append((q, a))
    return vid2qa

class FrozenFeatDataset(Dataset):
    def __init__(self, keys, feat_dict):
        self.keys = keys
        self.feat_dict = feat_dict
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        vid_id = self.keys[idx]
        return str(vid_id), self.feat_dict[vid_id]

# -----------------------------------------------------------------------------
# 3. Model Setup
# -----------------------------------------------------------------------------

class MockArgs:
    def __init__(self):
        self.model_name = DEBERTA_PATH
        self.load = CHECKPOINT_PATH
        self.scratch = False  
        
        self.freeze_bilm = True
        self.freeze_lm = True  
        self.freeze_mlm = True
        self.freeze_last = True
        self.freeze_adapter = False
        self.ft_ln = True      
        
        self.use_adapter = True
        self.adapter_dim = 512
        self.adapter_type = "normal"
        self.adapter_dropout = 0.1
        self.ds_factor_attn = 8 
        self.ds_factor_ff = 8   
        
        self.dropout = 0.1           
        self.attention_dropout = 0.1 
        self.cls_dropout = 0.1       
        
        self.max_tokens = MAX_TOKENS
        self.use_video = True
        
        # [注意] FrozenBiLM 默认使用 CLIP ViT-L/14，维度通常是 1024
        # 如果报错维度不匹配，请改回 768 或检查特征文件
        self.features_dim = 768 
        self.max_feats = 10   
        
        self.drop_path_rate = 0.1
        self.use_context = False
        self.use_cls_token = False
        self.n_ans = 0 

def setup_frozenbilm(device, is_master=True):
    args = MockArgs()
    if is_master: print(f"[Init] Tokenizer: {args.model_name} | Feat Dim: {args.features_dim}")
    
    tokenizer = get_tokenizer(args)
    model = build_model(args)
    model.to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        if is_master: print(f"[Init] Loading weights: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        
        state_dict = checkpoint.get('model', checkpoint)
        new_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
        
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if is_master and (missing or unexpected):
            print(f"[Info] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    model.eval()
    return model, tokenizer

# -----------------------------------------------------------------------------
# 4. Attack Logic (Batched)
# -----------------------------------------------------------------------------

def run_feature_attack(
    clean_feats, qa_list, model, tokenizer, config, device
):
    model.eval()
    
    questions = [x[0] for x in qa_list]
    answers = [x[1] for x in qa_list]
    
    # Prompt for MSRVTT
    prompts = [f"{q} Answer: [MASK]." for q in questions]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True, max_length=MAX_TOKENS).to(device)
    all_input_ids = inputs.input_ids
    all_attention_mask = inputs.attention_mask
    
    all_target_ids = []
    valid_indices = [] 
    
    for i, ans in enumerate(answers):
        ans_tokens = tokenizer(" " + ans, add_special_tokens=False)["input_ids"]
        if len(ans_tokens) > 0:
            all_target_ids.append(ans_tokens[0])
            valid_indices.append(i)
    
    if not valid_indices:
        return None, 0.0, 0.0, 0.0, 0.0 
        
    all_target_ids = torch.tensor(all_target_ids, device=device)
    all_input_ids = all_input_ids[valid_indices]
    all_attention_mask = all_attention_mask[valid_indices]
    
    TOTAL_QA = len(valid_indices)
    T = clean_feats.shape[1]
    
    mask_token_id = tokenizer.mask_token_id
    mask_indices = (all_input_ids == mask_token_id).nonzero(as_tuple=True)
    
    global_mask_pos = torch.zeros(TOTAL_QA, dtype=torch.long, device=device)
    # Handle potentially multiple masks? Just take the last/first valid one per row
    # Assuming one mask per prompt
    global_mask_pos[mask_indices[0]] = mask_indices[1]

    # --- PGD Init ---
    video_orig = clean_feats.detach().clone().to(device)
    delta = torch.zeros_like(video_orig).uniform_(-config['eps'], config['eps']).to(device)
    delta.requires_grad = True
    
    best_loss = -999.0
    best_adv_feat = video_orig + delta
    
    def run_batched_forward(video_feat, is_backward=False):
        total_loss_accum = 0.0
        all_logits = []
        
        num_batches = (TOTAL_QA + QA_BATCH_SIZE - 1) // QA_BATCH_SIZE
        
        for i in range(num_batches):
            start = i * QA_BATCH_SIZE
            end = min((i + 1) * QA_BATCH_SIZE, TOTAL_QA)
            curr_bs = end - start
            
            b_input = all_input_ids[start:end]
            b_mask = all_attention_mask[start:end]
            b_target = all_target_ids[start:end]
            b_mpos = global_mask_pos[start:end]
            
            b_vid = video_feat.expand(curr_bs, -1, -1)
            b_vid_mask = get_mask(torch.tensor([T]*curr_bs), T).to(device)
            
            outputs = model(
                video=b_vid, 
                video_mask=b_vid_mask, 
                input_ids=b_input, 
                attention_mask=b_mask
            )
            logits = outputs['logits']
            
            gather_idx = (T + b_mpos).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
            target_logits = torch.gather(logits, 1, gather_idx).squeeze(1)
            
            loss = F.cross_entropy(target_logits, b_target)
            scaled_loss = loss / num_batches 
            
            if is_backward:
                scaled_loss.backward()
            
            total_loss_accum += loss.item() 
            
            if not is_backward:
                all_logits.append(target_logits.detach())
                
        return total_loss_accum / num_batches, all_logits

    # --- Eval Clean ---
    with torch.no_grad():
        _, clean_logits_list = run_batched_forward(video_orig, is_backward=False)
        clean_logits_all = torch.cat(clean_logits_list, dim=0)
        acc1_c, acc10_c = accuracy(clean_logits_all, all_target_ids, topk=(1, 10))

    # --- PGD Loop ---
    for step in range(config['steps']):
        model.zero_grad()
        if delta.grad is not None: delta.grad.zero_()
        
        loss_val, _ = run_batched_forward(video_orig + delta, is_backward=True)
        
        with torch.no_grad():
            if delta.grad is not None:
                delta.data = delta.data + config['alpha'] * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -config['eps'], config['eps'])
            
            if loss_val > best_loss:
                best_loss = loss_val
                best_adv_feat = (video_orig + delta).detach().clone()

    # --- Eval Adv ---
    with torch.no_grad():
        _, adv_logits_list = run_batched_forward(best_adv_feat, is_backward=False)
        adv_logits_all = torch.cat(adv_logits_list, dim=0)
        acc1_a, acc10_a = accuracy(adv_logits_all, all_target_ids, topk=(1, 10))

    return best_adv_feat.cpu(), acc1_c, acc10_c, acc1_a, acc10_a

# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------

def main():
    local_rank, rank, world_size, device, is_master = init_dist()
    
    if is_master:
        print("="*60)
        print(" FrozenBiLM Feature Attack - MSRVTT")
        print(f" Features: {CLEAN_FEATS_PATH}")
        print(f" QA File:  {ANS_CSV}")
        print(f" Output:   {OUT_DIR}")
        print("="*60)
        os.makedirs(FEATS_SAVE_DIR, exist_ok=True)
    
    model, tokenizer = setup_frozenbilm(device, is_master)
    
    if is_master: print("[Data] Loading Features...")
    try:
        full_feats = torch.load(CLEAN_FEATS_PATH, map_location='cpu')
    except Exception as e:
        if is_master: print(f"[Error] Load features failed: {e}")
        return

    if is_master: print(f"[Data] Loading QA: {ANS_CSV}")
    vid2qa = load_qa_from_csv(ANS_CSV)
    
    valid_keys = []
    for k in full_feats.keys():
        k_str = str(k)
        found = False
        if k_str in vid2qa: found = True
        else:
            num = re.sub(r"\D", "", k_str)
            if num and num in vid2qa: found = True
        
        if found:
            # Range check logic if needed
            valid_keys.append(k)
            
    if is_master: print(f"[Data] Videos found: {len(valid_keys)}")
    if not valid_keys: return

    dataset = FrozenFeatDataset(valid_keys, full_feats)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=lambda x:x)
    
    attack_config = {'eps': FEAT_EPS, 'alpha': FEAT_ALPHA, 'steps': FEAT_STEPS}
    results = []
    
    for batch in tqdm(loader, desc=f"[R{rank}]", disable=(rank!=0)):
        vid_id_raw, feat_raw = batch[0]
        vid_str = str(vid_id_raw)
        
        feat = feat_raw.float()
        if feat.dim() == 2: feat = feat.unsqueeze(0)
        
        qa_list = vid2qa.get(vid_str, [])
        if not qa_list: qa_list = vid2qa.get(re.sub(r"\D", "", vid_str), [])
        if not qa_list: continue
        
        try:
            adv, c1, c10, a1, a10 = run_feature_attack(
                feat, qa_list, model, tokenizer, attack_config, device
            )
            
            if adv is None: continue
            
            save_name = f"{vid_str}.npy"
            np.save(os.path.join(FEATS_SAVE_DIR, save_name), adv.squeeze(0).numpy())
            
            results.append({
                'video_id': vid_str,
                'clean_acc1': c1, 'clean_acc10': c10,
                'adv_acc1': a1, 'adv_acc10': a10,
                'num_qa': len(qa_list)
            })
            
        except Exception as e:
            print(f"[R{rank}] Error {vid_str}: {e}")
            continue

    # Aggregate
    dist.barrier()
    all_res = [None]*world_size
    dist.all_gather_object(all_res, results)
    
    if is_master:
        flat = [i for s in all_res for i in s]
        unique = {r['video_id']: r for r in flat}.values()
        
        total_qa = sum(r['num_qa'] for r in unique)
        if total_qa > 0:
            c1 = sum(r['clean_acc1']*r['num_qa'] for r in unique)/total_qa*100
            c10 = sum(r['clean_acc10']*r['num_qa'] for r in unique)/total_qa*100
            a1 = sum(r['adv_acc1']*r['num_qa'] for r in unique)/total_qa*100
            a10 = sum(r['adv_acc10']*r['num_qa'] for r in unique)/total_qa*100
            
            print("\n" + "="*40)
            print(f" MSRVTT RESULTS (N={total_qa})")
            print(f" Clean: {c1:.2f} / {c10:.2f}")
            print(f" Adv:   {a1:.2f} / {a10:.2f}")
            print("="*40)
            
            ts = time.strftime('%Y%m%d-%H%M%S')
            csv_path = os.path.join(OUT_DIR, f"res_msrvtt_{ts}.csv")
            with open(csv_path, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=list(unique)[0].keys())
                writer.writeheader()
                writer.writerows(unique)
                
            print("Merging features...")
            merged = {}
            for f in os.listdir(FEATS_SAVE_DIR):
                if f.endswith('.npy'):
                    k = f.replace('.npy','')
                    try: k=int(k) 
                    except: pass
                    merged[k] = torch.from_numpy(np.load(os.path.join(FEATS_SAVE_DIR, f)))
            torch.save(merged, os.path.join(OUT_DIR, "adv_feats_merged.pth"))
            
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
