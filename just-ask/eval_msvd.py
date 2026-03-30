#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSVD 对抗攻击评估脚本 (Final Enhanced)
功能：
1. 合并 features_raw 下分散的 .pth 文件
2. 加载 test.csv 和 MMT 模型
3. 计算攻击后的 Top-1, Top-5, Top-10 Accuracy
"""

import os
import sys
import glob
import json
import csv
import torch
import numpy as np
from tqdm import tqdm

# ================= 配置区域 (必须与攻击脚本一致) =================
MAX_FEATS = 20   # 关键参数: 必须与 run_msvd_all_in_one.py 中的设置一致
QMAX_WORDS = 20

# 路径设置
JUSTASK_ROOT = "/data2/codefile/fjh/just-ask"
PROJ_ROOT = "/data2/codefile/fjh"
MSVD_QA_ROOT = "/data0/data/MSVD-QA"

if PROJ_ROOT not in sys.path: sys.path.insert(0, PROJ_ROOT)
if JUSTASK_ROOT not in sys.path: sys.path.insert(0, JUSTASK_ROOT)

ANS_CSV       = os.path.join(MSVD_QA_ROOT, "test.csv")
VOCAB_PATH    = os.path.join(MSVD_QA_ROOT, "vocab.json")
PRETRAIN_PATH = "/data0/pretrained/justask_checkpoints/ckpt_ft_msvdqa.pth"
BERT_PATH     = "/data0/pretrained/distilbert-base-uncased"

# 数据目录
RESULTS_BASE_DIR = "/data2/codefile/fjh/data_msvd/" 
OUT_DIR = os.path.join(RESULTS_BASE_DIR, "pixel_attack_no_norm")
RAW_ADV_FEATURES_DIR = os.path.join(OUT_DIR, "features_raw")
OUTPUT_PTH_PATH = os.path.join(OUT_DIR, "s3d_adv_features_msvd.pth")

# ================= 1. 合并特征 =================
def merge_features():
    print(">>> [Step 1] Merging Features...")
    pth_files = glob.glob(os.path.join(RAW_ADV_FEATURES_DIR, "*.pth"))
    if not pth_files:
        if os.path.exists(OUTPUT_PTH_PATH):
            print(f"Features raw files not found, but merged file exists at {OUTPUT_PTH_PATH}. Loading directly.")
            return torch.load(OUTPUT_PTH_PATH, map_location='cpu')
        print("❌ No feature files found!")
        sys.exit(1)
        
    merged = {}
    print(f"Found {len(pth_files)} feature files. Merging...")
    
    for p in tqdm(pth_files):
        try:
            vid = os.path.splitext(os.path.basename(p))[0]
            # 加载并转为 CPU，节省显存
            feat = torch.load(p, map_location='cpu')
            merged[vid] = feat
        except Exception as e:
            print(f"Error loading {p}: {e}")
            
    print(f"✅ Saving merged file to {OUTPUT_PTH_PATH}")
    torch.save(merged, OUTPUT_PTH_PATH)
    print(f"Total videos in merged file: {len(merged)}")
    return merged

# ================= 2. 评估逻辑 =================
def run_evaluation():
    print("\n>>> [Step 2] Evaluating Accuracy on Test Set...")
    
    # 依赖导入
    try:
        from model.multimodal_transformer import MMT_VideoQA
        from transformers import DistilBertTokenizer, BertTokenizer
    except ImportError:
        print("❌ Import failed. Check PYTHONPATH.")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载特征
    if not os.path.exists(OUTPUT_PTH_PATH):
        feats = merge_features()
    else:
        print(f"Loading features from {OUTPUT_PTH_PATH}...")
        feats = torch.load(OUTPUT_PTH_PATH, map_location='cpu')
    
    # 2. 加载 QA (只加载 test.csv)
    print("Loading Test QA...")
    qa_data = []
    
    try: f = open(ANS_CSV, "r", encoding='utf-8-sig')
    except: f = open(ANS_CSV, "r", encoding='utf-8')
    
    with f:
        rdr = csv.DictReader(f)
        for row in rdr:
            vid = row.get("video_id") or row.get("video") or row.get("id")
            if not vid: continue
            q = row.get("question")
            a = row.get("answer")
            if q and a:
                import re
                clean_id = re.sub(r"\D", "", str(vid))
                qa_data.append({'vid': clean_id, 'q': q.strip(), 'a': a.strip()})
    
    print(f"Test QA pairs: {len(qa_data)}")

    # 3. 加载模型
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    except:
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
        
    with open(VOCAB_PATH, "r") as f: a2id = json.load(f)
    id2a = {v:k for k,v in a2id.items()}
    
    # 准备 Answer Embeddings
    answers = [id2a[i] for i in range(len(id2a))]
    a2v = tokenizer(answers, add_special_tokens=True, max_length=10, 
                   padding="max_length", truncation=True, return_tensors="pt").input_ids.long().to(device)
    
    mmt = MMT_VideoQA(feature_dim=1024, word_dim=768, N=2, d_model=512, d_ff=2048, 
                      h=8, dropout=0.1, T=MAX_FEATS, Q=QMAX_WORDS, baseline=0).to(device)
    
    # 加载权重
    if os.path.exists(PRETRAIN_PATH):
        sd = torch.load(PRETRAIN_PATH, map_location="cpu")
        if 'model_state_dict' in sd: sd = sd['model_state_dict']
        mmt.load_state_dict({k.replace('module.', ''): v for k,v in sd.items()}, strict=False)
        
    mmt.eval()
    with torch.no_grad(): mmt._compute_answer_embedding(a2v)
    
    # 4. 批量推理
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    total = 0
    BATCH_SIZE = 128
    
    print("Start Inference...")
    for i in tqdm(range(0, len(qa_data), BATCH_SIZE)):
        batch = qa_data[i:i+BATCH_SIZE]
        
        fs, qs, ans = [], [], []
        for item in batch:
            # 获取特征
            f = feats.get(item['vid'])
            if f is None:
                continue
                
            if not isinstance(f, torch.Tensor): f = torch.from_numpy(f)
            f = f.float()
            
            # 维度修正
            if f.dim() == 1: f = f.unsqueeze(0) 
            
            cur_t = f.shape[0]
            if cur_t < MAX_FEATS:
                f = torch.cat([f, f[-1:].expand(MAX_FEATS-cur_t, -1)])
            elif cur_t > MAX_FEATS:
                idx = torch.linspace(0, cur_t-1, MAX_FEATS).long()
                f = f.index_select(0, idx)
                
            fs.append(f)
            qs.append(item['q'])
            ans.append(a2id.get(item['a'], 0)) 
            
        if not fs: continue
        
        # 构造 Batch
        fs_tensor = torch.stack(fs).to(device) # [B, 20, 1024]
        q_ids = tokenizer(qs, padding="max_length", truncation=True, 
                         max_length=QMAX_WORDS, return_tensors="pt").input_ids.to(device)
        ans_tensor = torch.tensor(ans).to(device)
        
        with torch.no_grad():
            text_mask = (q_ids > 0).float()
            video_mask = torch.ones(fs_tensor.shape[0], MAX_FEATS).to(device)
            
            logits = mmt(fs_tensor, q_ids, text_mask=text_mask, video_mask=video_mask)
            
            # --- 计算 Top-K ---
            # 获取 Top-10 预测
            _, top10_indices = logits.topk(10, dim=1) # [B, 10]
            
            # 扩展真实标签用于广播对比 [B, 1] -> [B, 10]
            targets_expanded = ans_tensor.view(-1, 1).expand_as(top10_indices)
            
            # 匹配矩阵 [B, 10] (True/False)
            matches = (top10_indices == targets_expanded)
            
            # 累计
            # Top-1: 检查第1列
            correct_1 += matches[:, 0].sum().item()
            
            # Top-5: 检查前5列
            correct_5 += matches[:, :5].sum().item()
            
            # Top-10: 检查前10列
            correct_10 += matches[:, :10].sum().item()
            
            total += len(qs)
            
    # 5. 打印结果
    print("\n" + "="*40)
    print("📊 Evaluation Results (Pixel Attack)")
    print("="*40)
    if total > 0:
        acc1 = correct_1 / total * 100
        acc5 = correct_5 / total * 100
        acc10 = correct_10 / total * 100
        
        print(f"Total Evaluated: {total}")
        print("-" * 20)
        print(f"🏆 Top-1  Accuracy: {acc1:.2f}%")
        print(f"🥈 Top-5  Accuracy: {acc5:.2f}%")
        print(f"🥉 Top-10 Accuracy: {acc10:.2f}%")
        print("="*40)
    else:
        print("⚠️ No samples evaluated!")

if __name__ == "__main__":
    run_evaluation()
