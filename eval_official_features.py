#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import DistilBertTokenizer, BertTokenizer
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 1. 配置与路径
# -----------------------------------------------------------------------------
JUSTASK_ROOT = "/data2/codefile/fjh/just-ask"
PROJ_ROOT = "/data2/codefile/fjh"
if PROJ_ROOT not in sys.path: sys.path.insert(0, PROJ_ROOT)
if JUSTASK_ROOT not in sys.path: sys.path.insert(0, JUSTASK_ROOT)

try:
    from model.multimodal_transformer import MMT_VideoQA
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# [关键输入] 官方特征文件路径
OFFICIAL_FEAT_PATH = "/data0/data/msrvtt/MSRVTT-QA/s3d.pth"

# 其他路径
ANS_CSV = os.path.join(JUSTASK_ROOT, "MSRVTT-QA/test.csv")
VOCAB_PATH = os.path.join(JUSTASK_ROOT, "MSRVTT-QA/vocab.json")
PRETRAIN_PATH = "/data0/pretrained/justask_checkpoints/ckpt_ft_msrvttqa.pth"
BERT_PATH = "/data0/pretrained/distilbert-base-uncased"

# 参数
MAX_FEATS = 20
QMAX_WORDS = 20
AMAX_WORDS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# 2. 辅助函数
# -----------------------------------------------------------------------------
def get_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    rng = torch.arange(max_len, device=lengths.device).expand(lengths.size(0), max_len)
    return rng < lengths.unsqueeze(1)

def encode_questions(tokenizer, questions, max_len, device):
    enc = tokenizer(questions, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return enc.input_ids.to(device)

def compute_a2v(vocab_path, bert_tokenizer, amax_words):
    with open(vocab_path, "r") as f: a2id = json.load(f)
    id2a = {v: k for k, v in a2id.items()}
    answers = [id2a[i] for i in range(max(id2a.keys()) + 1)]
    enc = bert_tokenizer(answers, add_special_tokens=True, max_length=amax_words,
                         padding="max_length", truncation=True, return_tensors="pt")
    return a2id, enc["input_ids"].long()

def load_qa_data(csv_path):
    data = []
    if not os.path.exists(csv_path): return data
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            vid = row.get("video_id") or row.get("video")
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            t = row.get("type", "default")
            if vid and q and a:
                # 统一 ID 格式，去除 .mp4 后缀
                base = re.sub(r"\.mp4$", "", vid, flags=re.IGNORECASE)
                data.append({'vid': base, 'q': q, 'a': a, 'type': t})
    return data

# -----------------------------------------------------------------------------
# 3. 主程序
# -----------------------------------------------------------------------------
def main():
    print("="*60)
    print(f"Evaluating Official Features (Fix Integer Key Match)")
    print(f"File: {OFFICIAL_FEAT_PATH}")
    print("="*60)

    # 1. 加载特征
    if not os.path.exists(OFFICIAL_FEAT_PATH):
        print("Error: Feature file not found!")
        return

    print("Loading feature dictionary...")
    feats_dict = torch.load(OFFICIAL_FEAT_PATH, map_location='cpu')
    print(f"Loaded {len(feats_dict)} features.")
    
    # 打印 Key 类型以确诊
    first_key = list(feats_dict.keys())[0]
    print(f"Sample key: {first_key}, Type: {type(first_key)}")

    # 2. 准备模型
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    except:
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    
    a2id, a2v = compute_a2v(VOCAB_PATH, tokenizer, AMAX_WORDS)
    a2v = a2v.to(DEVICE)

    model = MMT_VideoQA(
        feature_dim=1024, word_dim=768, N=2, d_model=512, d_ff=2048, 
        h=8, dropout=0.1, T=MAX_FEATS, Q=QMAX_WORDS, baseline=0
    ).to(DEVICE)

    if os.path.exists(PRETRAIN_PATH):
        print(f"Loading QA Model: {PRETRAIN_PATH}")
        state = torch.load(PRETRAIN_PATH, map_location="cpu")
        if 'model_state_dict' in state: state = state['model_state_dict']
        new_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        model.load_state_dict(new_sd, strict=False)
    
    with torch.no_grad(): model._compute_answer_embedding(a2v)
    model.eval()

    # 3. 加载 QA 数据
    qa_data = load_qa_data(ANS_CSV)
    print(f"Total QA pairs: {len(qa_data)}")

    # 4. 评测循环
    correct_1 = 0
    correct_10 = 0
    total = 0
    missing_count = 0

    BATCH_SIZE = 128
    
    for i in tqdm(range(0, len(qa_data), BATCH_SIZE), desc="Evaluating"):
        batch_items = qa_data[i : i + BATCH_SIZE]
        
        questions = []
        answers_indices = []
        feat_batch_list = []
        
        for idx, item in enumerate(batch_items):
            vid_str = item['vid'] # e.g. "video1001" or "1001"
            
            # 提取纯数字部分
            num_part = re.sub(r"\D", "", vid_str)
            
            feat = None
            
            # --- 核心修复：尝试多种 Key 格式匹配 ---
            # 1. 尝试匹配整数 Key (e.g. 1001)
            if num_part and int(num_part) in feats_dict:
                feat = feats_dict[int(num_part)]
                
            # 2. 尝试匹配纯数字字符串 Key (e.g. "1001")
            elif num_part and num_part in feats_dict:
                feat = feats_dict[num_part]
                
            # 3. 尝试匹配原始字符串 (e.g. "video1001")
            elif vid_str in feats_dict:
                feat = feats_dict[vid_str]
            
            if feat is None:
                missing_count += 1
                continue
                
            # 处理特征维度
            if not isinstance(feat, torch.Tensor):
                feat = torch.from_numpy(feat)
            
            # 确保是 Float Tensor
            feat = feat.float()
            
            # 维度 [T, 1024]
            if feat.dim() == 3: # [1, T, 1024]
                feat = feat.squeeze(0)
            
            # 采样/补齐到 MAX_FEATS (20)
            T_feat = feat.shape[0]
            if T_feat >= MAX_FEATS:
                samp_idx = torch.linspace(0, T_feat - 1, MAX_FEATS).long()
                feat = feat[samp_idx]
            else:
                padding = feat[-1:].expand(MAX_FEATS - T_feat, -1)
                feat = torch.cat([feat, padding], dim=0)
            
            feat_batch_list.append(feat)
            questions.append(item['q'])
            answers_indices.append(a2id.get(item['a'], 0))

        if not questions:
            continue

        # Stack
        feat_batch = torch.stack(feat_batch_list).to(DEVICE) # [B, 20, 1024]
        q_ids = encode_questions(tokenizer, questions, QMAX_WORDS, DEVICE)
        a_ids = torch.tensor(answers_indices, dtype=torch.long, device=DEVICE)
        
        B = len(questions)
        
        # Masks
        text_mask = (q_ids > 0).float()
        video_len = torch.full((B,), MAX_FEATS, dtype=torch.long, device=DEVICE)
        video_mask = get_mask(video_len, MAX_FEATS)
        
        with torch.no_grad():
            logits = model(feat_batch, q_ids, text_mask=text_mask, video_mask=video_mask)
            
            # Top-1
            preds = logits.argmax(dim=1)
            correct_1 += (preds == a_ids).sum().item()
            
            # Top-10
            _, top10 = logits.topk(10, dim=1)
            correct_10 += (top10 == a_ids.unsqueeze(1)).any(dim=1).sum().item()
            
            total += B

    print("\n" + "="*60)
    print(f"Processed QA Pairs: {total}")
    print(f"Missing Features  : {missing_count}")
    if total > 0:
        print(f"Official Feat Top-1 Acc : {correct_1 / total * 100:.2f}%")
        print(f"Official Feat Top-10 Acc: {correct_10 / total * 100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
