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
# 1. Configuration & Paths
# -----------------------------------------------------------------------------

# FrozenBiLM 项目路径
FROZEN_ROOT = "/data2/codefile/fjh/FrozenBilm"
if FROZEN_ROOT not in sys.path: sys.path.insert(0, FROZEN_ROOT)

print(f"[Debug] Python Path: {sys.path}")
print(f"[Debug] Checking contents of {FROZEN_ROOT}:")
try:
    print(os.listdir(FROZEN_ROOT))
except Exception as e:
    print(f"  Cannot list dir: {e}")

try:
    from model import build_model, get_tokenizer
    from util.misc import get_mask
    print("✅ Successfully imported FrozenBiLM modules.")
except Exception as e:
    print("\n" + "!"*60)
    print(f"❌ 导入失败！具体报错如下：")
    print(f"{e}")
    print("!"*60 + "\n")
    import traceback
    traceback.print_exc() # 打印完整报错堆栈
    sys.exit(1)

# --- 输入路径 ---
CLEAN_FEATS_PATH = "/data2/codefile/fjh/FrozenBilm/clipvitl14.pth"
ANS_CSV = "/data0/data/MSVD-QA/test.csv"

# 模型权重
DEBERTA_PATH = "/data0/pretrained/deberta-v2-xlarge/" 
CHECKPOINT_PATH = "/data0/pretrained/FrozenBilm/checkpoints/frozenbilm_msvd.pth"

# --- 输出路径 ---
RESULTS_BASE_DIR = "/data2/codefile/fjh/FrozenBilm/results_msvd"
OUT_DIR = os.path.join(RESULTS_BASE_DIR, "feats_frozenbilm_10")
FEATS_SAVE_DIR = os.path.join(OUT_DIR, "adv_features_npy")

# --- 攻击参数 ---
FEAT_EPS = 0.5      
FEAT_ALPHA = 0.1  
FEAT_STEPS = 10       
RESTARTS = 1

# [显存控制] QA 分批大小
QA_BATCH_SIZE = 4  

RANGE_LOW, RANGE_HIGH = 0, 9999
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
    vid2qa = {}
    if not os.path.exists(csv_path): return vid2qa
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            vid = row.get("video_id") or row.get("video")
            q = row.get("question")
            a = row.get("answer")
            if not vid or not q or not a: continue
            
            base = re.sub(r".mp4$", "", vid, flags=re.IGNORECASE)
            vid2qa.setdefault(base, []).append((q, a))
            
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
        self.features_dim = 768
        self.max_feats = 10   
        
        self.drop_path_rate = 0.1
        self.use_context = False
        self.use_cls_token = False
        self.n_ans = 0 

def setup_frozenbilm(device):
    args = MockArgs()
    tokenizer = get_tokenizer(args)
    model = build_model(args)
    model.to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    model.eval()
    return model, tokenizer

# -----------------------------------------------------------------------------
# 4. Attack Logic (QA Batching ONLY)
# -----------------------------------------------------------------------------

def run_feature_attack(
    clean_feats, qa_list, model, tokenizer, config, device
):
    model.eval()
    
    # 1. 准备文本输入
    questions = [x[0] for x in qa_list]
    answers = [x[1] for x in qa_list]
    
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
    global_mask_pos[mask_indices[0]] = mask_indices[1]

    # --- PGD Attack ---
    video_orig = clean_feats.detach().clone().to(device)
    delta = torch.zeros_like(video_orig).uniform_(-config['eps'], config['eps']).to(device)
    delta.requires_grad = True
    
    best_loss = -999.0
    best_adv_feat = video_orig + delta
    
    # ------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------
    def run_batched_forward(video_feat, is_backward=False):
        total_loss_accum = 0.0
        all_logits = []
        
        num_batches = (TOTAL_QA + QA_BATCH_SIZE - 1) // QA_BATCH_SIZE
        
        for i in range(num_batches):
            start = i * QA_BATCH_SIZE
            end = min((i + 1) * QA_BATCH_SIZE, TOTAL_QA)
            curr_bs = end - start
            
            b_input_ids = all_input_ids[start:end]
            b_att_mask = all_attention_mask[start:end]
            b_target_ids = all_target_ids[start:end]
            b_mask_pos = global_mask_pos[start:end]
            
            b_video = video_feat.expand(curr_bs, -1, -1)
            b_video_mask = get_mask(torch.tensor([T]*curr_bs), T).to(device)
            
            outputs = model(
                video=b_video, 
                video_mask=b_video_mask, 
                input_ids=b_input_ids, 
                attention_mask=b_att_mask
            )
            logits = outputs['logits']
            
            gather_idx = (T + b_mask_pos).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
            target_logits = torch.gather(logits, 1, gather_idx).squeeze(1)
            
            loss = F.cross_entropy(target_logits, b_target_ids)
            scaled_loss = loss / num_batches 
            
            if is_backward:
                scaled_loss.backward()
            
            total_loss_accum += loss.item() 
            
            if not is_backward:
                all_logits.append(target_logits.detach())
                
        return total_loss_accum / num_batches, all_logits

    # --- Step 0: Clean Eval ---
    _, clean_logits_list = run_batched_forward(video_orig, is_backward=False)
    clean_logits_all = torch.cat(clean_logits_list, dim=0)
    acc1_c, acc10_c = accuracy(clean_logits_all, all_target_ids, topk=(1, 10))

    # --- Step 1: PGD Loop ---
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

    # --- Step 2: Adv Eval ---
    with torch.no_grad():
        _, adv_logits_list = run_batched_forward(best_adv_feat, is_backward=False)
        adv_logits_all = torch.cat(adv_logits_list, dim=0)
        acc1_a, acc10_a = accuracy(adv_logits_all, all_target_ids, topk=(1, 10))

    # [修复点] 去掉了 [0]
    return best_adv_feat.cpu(), acc1_c, acc10_c, acc1_a, acc10_a

# -----------------------------------------------------------------------------
# 5. Main Loop
# -----------------------------------------------------------------------------

def main():
    local_rank, rank, world_size, device, is_master = init_dist()
    
    if is_master:
        print("="*60)
        print(" FrozenBiLM Feature-Space Attack (Memory Optimized)")
        print(f" Clean Features: {CLEAN_FEATS_PATH}")
        print(f" QA CSV: {ANS_CSV}")
        print(f" QA Batch Size: {QA_BATCH_SIZE}")
        print("="*60)
        os.makedirs(FEATS_SAVE_DIR, exist_ok=True)
    
    # 1. Load Model
    model, tokenizer = setup_frozenbilm(device)
    
    # 2. Load Features
    if is_master: print("[Data] Loading CLIP features...")
    try:
        full_feats = torch.load(CLEAN_FEATS_PATH, map_location='cpu')
    except Exception as e:
        if is_master: print(f"[Error] Failed to load features: {e}")
        return

    # 3. Filter Data
    if is_master: print(f"[Data] Loading QA from {ANS_CSV}...")
    vid2qa = load_qa_from_csv(ANS_CSV)
    
    valid_keys = []
    for k in full_feats.keys():
        k_str = str(k)
        if k_str in vid2qa:
            try:
                num = int(re.sub(r"\D", "", k_str))
                if RANGE_LOW <= num <= RANGE_HIGH:
                    valid_keys.append(k)
            except: pass
        else:
            num_key = re.sub(r"\D", "", k_str)
            if num_key and num_key in vid2qa:
                if RANGE_LOW <= int(num_key) <= RANGE_HIGH:
                    valid_keys.append(k)
            
    if is_master: print(f"[Data] Target videos: {len(valid_keys)}")
    if len(valid_keys) == 0: return

    # 4. Dataset & Loader
    dataset = FrozenFeatDataset(valid_keys, full_feats)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=lambda x:x)
    
    attack_config = {'eps': FEAT_EPS, 'alpha': FEAT_ALPHA, 'steps': FEAT_STEPS}
    results = []
    
    # 5. Attack Loop
    for batch in tqdm(loader, desc=f"[Rank {rank}]", disable=(rank!=0)):
        vid_id_raw, feat_raw = batch[0]
        vid_str = str(vid_id_raw)
        
        feat = feat_raw.float()
        if feat.dim() == 2: feat = feat.unsqueeze(0) 
        
        qa_list = vid2qa.get(vid_str, [])
        if not qa_list:
            qa_list = vid2qa.get(re.sub(r"\D", "", vid_str), [])
        
        if not qa_list: continue
        
        try:
            adv_feat, c1, c10, a1, a10 = run_feature_attack(
                feat, qa_list, model, tokenizer, attack_config, device
            )
            
            if adv_feat is None: continue 
            
            # Save
            save_name = f"{vid_str}.npy"
            np.save(os.path.join(FEATS_SAVE_DIR, save_name), adv_feat.squeeze(0).numpy())
            
            results.append({
                'video_id': vid_str,
                'clean_acc1': c1, 'clean_acc10': c10,
                'adv_acc1': a1, 'adv_acc10': a10,
                'num_qa': len(qa_list)
            })
            
        except Exception as e:
            print(f"[Rank {rank}] Error {vid_str}: {e}")
            torch.cuda.empty_cache()
            continue

    # 6. Aggregate
    dist.barrier()
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)
    
    if is_master:
        flat_results = [r for sublist in all_results for r in sublist]
        unique_res = {}
        for r in flat_results:
            unique_res[r['video_id']] = r
        final_res = list(unique_res.values())
        
        total_qa = sum(r['num_qa'] for r in final_res)
        
        if total_qa > 0:
            c1_avg = sum(r['clean_acc1'] * r['num_qa'] for r in final_res) / total_qa * 100
            c10_avg = sum(r['clean_acc10'] * r['num_qa'] for r in final_res) / total_qa * 100
            a1_avg = sum(r['adv_acc1'] * r['num_qa'] for r in final_res) / total_qa * 100
            a10_avg = sum(r['adv_acc10'] * r['num_qa'] for r in final_res) / total_qa * 100
            
            print("\n" + "="*50)
            print(f" FINAL RESULTS (QA Count: {total_qa})")
            print("-" * 50)
            print(f" Clean | Top-1: {c1_avg:.2f}% | Top-10: {c10_avg:.2f}%")
            print(f" Adv   | Top-1: {a1_avg:.2f}% | Top-10: {a10_avg:.2f}%")
            print("="*50)
            
            ts = time.strftime('%Y%m%d-%H%M%S')
            csv_path = os.path.join(OUT_DIR, f"results_frozenbilm_{ts}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=final_res[0].keys())
                writer.writeheader()
                writer.writerows(final_res)
            print(f"Saved stats to {csv_path}")
            
            print("Merging features...")
            merged = {}
            for fname in os.listdir(FEATS_SAVE_DIR):
                if fname.endswith(".npy"):
                    k = fname.replace(".npy", "")
                    try: k_final = int(k)
                    except: k_final = k
                    merged[k_final] = torch.from_numpy(np.load(os.path.join(FEATS_SAVE_DIR, fname)))
            
            torch.save(merged, os.path.join(OUT_DIR, "adv_features_merged.pth"))
            print("Done.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()



