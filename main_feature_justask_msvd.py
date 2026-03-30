#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import re
import json
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import DistilBertTokenizer, BertTokenizer
from typing import List, Dict, Tuple, Optional

# -----------------------------------------------------------------------------
# 1. Configuration & Paths (MSVD Setup)
# -----------------------------------------------------------------------------

# 项目根目录
JUSTASK_ROOT = "/data2/codefile/fjh/just-ask"
PROJ_ROOT = "/data2/codefile/fjh"

if PROJ_ROOT not in sys.path: sys.path.insert(0, PROJ_ROOT)
if JUSTASK_ROOT not in sys.path: sys.path.insert(0, JUSTASK_ROOT)

try:
    from model.multimodal_transformer import MMT_VideoQA
except ImportError:
    print("Error: Could not import MMT_VideoQA. Check JUSTASK_ROOT path.")
    sys.exit(1)

# --- MSVD 数据路径 ---
DATA_BASE_DIR = "/data0/data/msvd/" 
MSVD_QA_ROOT = "/data0/data/MSVD-QA"

# [关键] 原始干净特征文件路径
# 注意：代码是对特征进行攻击，需要加载提取好的 .pth 特征文件
# 请确认你的 MSVD S3D 特征文件是否在这个路径，如果不是请修改此处！
CLEAN_FEATS_PATH = os.path.join(MSVD_QA_ROOT, "s3d.pth") 

# QA 标注文件 & 词表
ANS_CSV = os.path.join(MSVD_QA_ROOT, "test.csv")
VOCAB_PATH = os.path.join(MSVD_QA_ROOT, "vocab.json")

# 模型权重
PRETRAIN_PATH = "/data0/pretrained/justask_checkpoints/ckpt_ft_msvdqa.pth"
BERT_PATH = "/data0/pretrained/distilbert-base-uncased"
# S3D_WEIGHT_PATH 在特征攻击代码中不需要，除非你要现场提取特征

# --- 输出路径 ---
RESULTS_BASE_DIR = "/data2/codefile/fjh/data/"
OUT_DIR = os.path.join(RESULTS_BASE_DIR, "feats_justask_msvd_attack") # 修改输出目录名
FEATS_SAVE_DIR = os.path.join(OUT_DIR, "adv_features_npy")

# --- 模型参数 ---
# 【重要警告】: 请务必运行 check.py 检查 ckpt_ft_msvdqa.pth 的位置编码形状
# 如果是 [20, 512] 则设为 20，如果是 [40, 512] 则设为 40。
MAX_FEATS = 20  
FEATURE_DIM = 1024
WORD_DIM = 768
N_LAYERS = 2
EMBD_DIM = 512
FF_DIM = 2048
N_HEADS = 8
DROPOUT = 0.1
QMAX_WORDS = 20
AMAX_WORDS = 10

# --- PGD 攻击参数 ---
FEAT_EPS = 0.5     # 攻击阈值
FEAT_ALPHA = 0.1         # 步长
FEAT_STEPS = 10          # 迭代步数
RESTARTS = 1

# 视频ID过滤范围
RANGE_LOW, RANGE_HIGH = 0, 99999 # MSVD ID 范围

# -----------------------------------------------------------------------------
# 2. Utils & Setup
# -----------------------------------------------------------------------------

def init_dist():
    """初始化 DDP 环境"""
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

def get_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    rng = torch.arange(max_len, device=lengths.device).expand(lengths.size(0), max_len)
    return rng < lengths.unsqueeze(1)

def accuracy(output, target, topk=(1,)):
    """计算 Top-K 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if output.size(1) < maxk: maxk = output.size(1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            if k > maxk: k = maxk
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item() / batch_size)
        return res

def compute_a2v(vocab_path, bert_tokenizer, amax_words):
    with open(vocab_path, "r") as f: a2id = json.load(f)
    id2a = {v: k for k, v in a2id.items()}
    answers = [id2a[i] for i in range(max(id2a.keys()) + 1)]
    enc = bert_tokenizer(answers, add_special_tokens=True, max_length=amax_words,
                         padding="max_length", truncation=True, return_tensors="pt")
    return a2id, id2a, enc["input_ids"].long()

def encode_questions(tokenizer, questions: List[str], max_len: int, device: torch.device):
    enc = tokenizer(questions, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return enc.input_ids.to(device)

def load_qa_from_csv(csv_path: str):
    """
    针对 MSVD 格式进行优化
    MSVD CSV 常见列名: 'video_id', 'question', 'answer'
    Video ID 格式通常为: 'vid1', 'vid100' 或 直接数字
    """
    vid2qa = {}
    if not os.path.exists(csv_path): return vid2qa
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            # 兼容多种列名
            vid = row.get("video_id") or row.get("video") or row.get("id")
            if not vid: continue
            
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not q or not a: continue
            
            # 统一清理: vid123 -> 123
            base = str(vid).strip()
            num = re.sub(r"\D", "", base)
            
            # 存储映射：既存原始字符串，也存数字ID字符串，增加命中率
            if num:
                vid2qa.setdefault(num, []).append((q, a))
            vid2qa.setdefault(base, []).append((q, a))
            
    return vid2qa

class S3DFeatureDataset(Dataset):
    def __init__(self, feature_keys, features_dict):
        self.keys = feature_keys
        self.features = features_dict
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        vid_id = self.keys[idx]
        feat = self.features[vid_id]
        return str(vid_id), feat

# -----------------------------------------------------------------------------
# 3. Core Attack Logic
# -----------------------------------------------------------------------------

def run_direct_feature_attack(
    clean_feats, questions_ids, answer_ids, model, config, device
):
    model.eval()
    Q = questions_ids.size(0)

    # 1. 准备 Clean Input
    video_clean_1 = clean_feats.detach().to(device)
    T, D = video_clean_1.size(1), video_clean_1.size(2)
    video_clean_q = video_clean_1.expand(Q, T, D).contiguous()
    
    text_mask = (questions_ids > 0).float()
    video_len = torch.full((Q,), T, dtype=torch.long, device=device)
    video_mask = get_mask(video_len, T)
    
    # Clean Acc
    with torch.no_grad():
        logits_clean = model(video_clean_q, questions_ids, text_mask=text_mask, video_mask=video_mask)
        acc1_clean, acc10_clean = accuracy(logits_clean, answer_ids, topk=(1, 10))

    # 2. PGD Attack
    video_orig = video_clean_1.clone()
    best_loss = -float('inf')
    best_adv_feats = video_orig.clone()

    for r in range(config['restarts']):
        delta = torch.empty_like(video_orig).uniform_(-config['eps'], config['eps']).to(device)
        delta.requires_grad_(True)

        for t in range(config['steps']):
            video_adv_1 = video_orig + delta
            video_adv_q = video_adv_1.expand(Q, T, D).contiguous()
            
            logits = model(video_adv_q, questions_ids, text_mask=text_mask, video_mask=video_mask)
            loss = F.cross_entropy(logits, answer_ids)
            
            model.zero_grad()
            if delta.grad is not None: delta.grad.zero_()
            loss.backward()
            
            with torch.no_grad():
                delta.add_(config['alpha'] * delta.grad.sign())
                delta.clamp_(-config['eps'], config['eps'])
        
        # Eval current restart
        with torch.no_grad():
            final_adv_feats_1 = video_orig + delta
            final_adv_feats_q = final_adv_feats_1.expand(Q, T, D).contiguous()
            final_logits = model(final_adv_feats_q, questions_ids, text_mask=text_mask, video_mask=video_mask)
            final_loss = F.cross_entropy(final_logits, answer_ids).item()
            
            if final_loss > best_loss:
                best_loss = final_loss
                best_adv_feats = final_adv_feats_1.detach()

    return best_adv_feats.cpu(), acc1_clean, acc10_clean

# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------

def main():
    local_rank, rank, world_size, device, is_master = init_dist()

    if is_master:
        print("="*60)
        print(" MSVD-QA Feature-Space PGD Attack (Multi-GPU)")
        print(f" Feature File: {CLEAN_FEATS_PATH}")
        print(f" Output Dir:   {OUT_DIR}")
        print(f" Checkpoint:   {PRETRAIN_PATH}")
        print("="*60)
        os.makedirs(FEATS_SAVE_DIR, exist_ok=True)

    # --- Load Tokenizer ---
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    except:
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    
    a2id, id2a, a2v = compute_a2v(vocab_path=VOCAB_PATH, bert_tokenizer=tokenizer, amax_words=AMAX_WORDS)
    a2v = a2v.to(device)

    # --- Load Transformer Model ---
    if is_master: print(f"[Config] Initializing model with T={MAX_FEATS}...")
    model = MMT_VideoQA(
        feature_dim=FEATURE_DIM, word_dim=WORD_DIM, N=N_LAYERS, d_model=EMBD_DIM, d_ff=FF_DIM, 
        h=N_HEADS, dropout=DROPOUT, T=MAX_FEATS, Q=QMAX_WORDS, baseline=0
    ).to(device)

    if os.path.exists(PRETRAIN_PATH):
        state = torch.load(PRETRAIN_PATH, map_location="cpu")
        if 'model_state_dict' in state: state = state['model_state_dict']
        new_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        
        # 使用 strict=True 来捕捉 MAX_FEATS 设置错误的问题
        try:
            model.load_state_dict(new_sd, strict=True)
            if is_master: print(f"[Model] Loaded weights SUCCESSFULLY from {PRETRAIN_PATH}")
        except RuntimeError as e:
            if is_master:
                print(f"\n[CRITICAL ERROR] Model Load Failed!")
                print(f"Reason: {e}")
                print(f"提示: 请检查 MAX_FEATS ({MAX_FEATS}) 是否与 Checkpoint 中的 Positional Embedding 形状匹配。")
                print(f"请使用 check.py 脚本查看 {PRETRAIN_PATH} 的参数形状。\n")
            sys.exit(1)
    else:
        if is_master: print(f"[Error] Checkpoint not found: {PRETRAIN_PATH}")
        return

    with torch.no_grad(): model._compute_answer_embedding(a2v)
    model.eval()

    # --- Load Features ---
    if is_master: print(f"[Data] Loading MSVD feature dictionary from {CLEAN_FEATS_PATH}...")
    try:
        full_features_dict = torch.load(CLEAN_FEATS_PATH, map_location='cpu')
    except Exception as e:
        if is_master: 
            print(f"[Error] Failed to load features: {e}")
            print("请确认 CLEAN_FEATS_PATH 路径是否正确。")
        return

    # --- Filter Data (QA & Range) ---
    vid2qa = load_qa_from_csv(ANS_CSV)
    valid_keys = []
    
    # 筛选
    for key in full_features_dict.keys():
        key_str = str(key)
        # 尝试清理出数字部分
        num_key = re.sub(r"\D", "", key_str)
        
        # 只要能在 QA 字典里找到对应的 ID (无论是原始 key 还是纯数字)
        if (key_str in vid2qa) or (num_key and num_key in vid2qa):
            try:
                if num_key:
                    vid_num = int(num_key)
                    if RANGE_LOW <= vid_num <= RANGE_HIGH:
                        valid_keys.append(key)
            except:
                pass

    if is_master: print(f"[Data] {len(valid_keys)} videos ready for MSVD attack.")

    # --- DDP Dataset & Loader ---
    dataset = S3DFeatureDataset(valid_keys, full_features_dict)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=lambda x: x)

    attack_config = {
        'eps': FEAT_EPS, 'alpha': FEAT_ALPHA, 'steps': FEAT_STEPS, 
        'restarts': RESTARTS, 'max_feats': MAX_FEATS
    }

    results = []

    # --- Main Attack Loop ---
    for batch in tqdm(loader, desc=f"[Rank {rank}]", disable=(rank!=0)):
        vid_id_raw, feat_raw = batch[0]
        vid_str = str(vid_id_raw)
        
        # 1. 特征维度调整
        if isinstance(feat_raw, np.ndarray): feat_raw = torch.from_numpy(feat_raw)
        feat = feat_raw.float()
        
        if feat.dim() == 3: feat = feat.squeeze(0) 
        if feat.dim() == 1: feat = feat.unsqueeze(0) 
        
        # 2. 强制对齐到 MAX_FEATS (Padding or Sampling)
        K = feat.shape[0]
        T_target = MAX_FEATS
        
        if K >= T_target:
            idx = torch.linspace(0, K - 1, T_target).long()
            v = feat.index_select(0, idx)
        else:
            v = torch.cat([feat, feat[-1:].expand(T_target - K, feat.shape[1])], dim=0)
            
        clean_feat_input = v.unsqueeze(0) 
        
        # 3. 获取 QA
        # 尝试多种 key 查找方式 (vid1, 1, video1)
        qa_list = vid2qa.get(vid_str, [])
        if not qa_list:
            qa_list = vid2qa.get(re.sub(r"\D", "", vid_str), [])
        
        if not qa_list: continue

        questions_text = [q for q, _ in qa_list]
        answers_text = [a for _, a in qa_list]
        
        questions_ids = encode_questions(tokenizer, questions_text, QMAX_WORDS, device)
        answer_ids = torch.tensor([a2id.get(a, 0) for a in answers_text], dtype=torch.long, device=device)
        
        try:
            # 4. 执行攻击
            adv_feats, acc1_clean, acc10_clean = run_direct_feature_attack(
                clean_feat_input, questions_ids, answer_ids, model, attack_config, device
            )
            
            # 5. 验证攻击效果
            with torch.no_grad():
                Q, T, D = questions_ids.size(0), adv_feats.size(1), adv_feats.size(2)
                adv_feats_q = adv_feats.to(device).expand(Q, T, D).contiguous()
                text_mask = (questions_ids > 0).float()
                video_len = torch.full((Q,), T, dtype=torch.long, device=device)
                video_mask = get_mask(video_len, T)
                
                logits_adv = model(adv_feats_q, questions_ids, text_mask=text_mask, video_mask=video_mask)
                acc1_adv, acc10_adv = accuracy(logits_adv, answer_ids, topk=(1, 10))

            # 6. 保存
            save_name = f"{vid_str}.npy"
            np.save(os.path.join(FEATS_SAVE_DIR, save_name), adv_feats.squeeze(0).numpy())

            results.append({
                'video_id': vid_str, 
                'clean_acc1': acc1_clean, 'clean_acc10': acc10_clean,
                'adv_acc1': acc1_adv, 'adv_acc10': acc10_adv,
                'num_qa': len(qa_list)
            })
            
        except Exception as e:
            print(f"[Rank {rank}] Error {vid_str}: {e}")
            continue

    # --- 结果汇总 ---
    dist.barrier()
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)

    if is_master:
        print("\n[Done] Aggregating MSVD results...")
        flat_results = []
        for res_list in all_results: flat_results.extend(res_list)
        
        unique_results = []
        seen = set()
        for r in flat_results:
            if r['video_id'] not in seen:
                unique_results.append(r)
                seen.add(r['video_id'])

        total_qa = sum(r['num_qa'] for r in unique_results)
        
        if total_qa > 0:
            avg_c1 = sum(r['clean_acc1'] * r['num_qa'] for r in unique_results) / total_qa * 100
            avg_c10 = sum(r['clean_acc10'] * r['num_qa'] for r in unique_results) / total_qa * 100
            avg_a1 = sum(r['adv_acc1'] * r['num_qa'] for r in unique_results) / total_qa * 100
            avg_a10 = sum(r['adv_acc10'] * r['num_qa'] for r in unique_results) / total_qa * 100
            
            print("\n" + "="*50)
            print(f" MSVD FINAL REPORT (Total QA: {total_qa})")
            print("-"*50)
            print(f" Clean | Top-1: {avg_c1:.2f}% | Top-10: {avg_c10:.2f}%")
            print(f" Adv   | Top-1: {avg_a1:.2f}% | Top-10: {avg_a10:.2f}%")
            print("="*50 + "\n")
            
            ts = time.strftime('%Y%m%d-%H%M%S')
            csv_path = os.path.join(OUT_DIR, f"result_msvd_attack_{ts}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=unique_results[0].keys())
                writer.writeheader()
                writer.writerows(unique_results)
            print(f"CSV saved to: {csv_path}")
            
            # Merge
            merged_feats = {}
            for fname in os.listdir(FEATS_SAVE_DIR):
                if fname.endswith(".npy"):
                    vid_key = fname.replace(".npy", "")
                    arr = np.load(os.path.join(FEATS_SAVE_DIR, fname))
                    merged_feats[vid_key] = torch.from_numpy(arr)
            
            merge_path = os.path.join(OUT_DIR, "msvd_adv_features_merged.pth")
            torch.save(merged_feats, merge_path)
            print(f"[Info] Merged features saved to: {merge_path}")
            
        else:
            print("[Warning] No QA results gathered. Check if CSV IDs match Feature Keys.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
