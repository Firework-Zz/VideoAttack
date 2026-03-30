#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrozenBiLM Pixel Attack 评估脚本 (MSVD)
适配: 加载合并后的 .pth 文件，并计算 Top-1/5/10 准确率
"""

import sys
import os
import csv
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import re

# === [Patch] 修复 Numpy 版本兼容性问题 ===
if not hasattr(np, 'int'): np.int = int

# 1. 环境设置 (请确保在 FrozenBiLM 根目录下运行，或修改 sys.path)
FROZEN_ROOT = "/data2/codefile/fjh/FrozenBilm"
if FROZEN_ROOT not in sys.path: sys.path.insert(0, FROZEN_ROOT)

try:
    from model import build_model, get_tokenizer
    from util.misc import get_mask
except ImportError:
    print(f"[ERROR] 无法导入 FrozenBiLM 模块。请检查 FROZEN_ROOT: {FROZEN_ROOT}")
    sys.exit(1)

# -----------------------------------------------------------
# 配置参数
# -----------------------------------------------------------
class EvalConfig:
    def __init__(self):
        # === [路径配置] ===
        # 1. 你的对抗特征文件 (合并后的大文件)
        self.adv_features_path = "/data2/codefile/fjh/FrozenBilm/attack_results_msvd/clip_pixel_attack/frozenbilm_adv_clip_msvd.pth"
        
        # 2. MSVD 测试集 QA
        self.qa_csv_path = "/data0/data/MSVD-QA/test.csv"
        
        # 3. Checkpoint (MSVD 训练好的权重)
        self.load = "/data0/pretrained/FrozenBilm/checkpoints/frozenbilm_msvd.pth"
        
        # 4. 底座模型路径
        self.model_name = "/data0/pretrained/deberta-v2-xlarge" 
        
        # === [模型参数] ===
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.features_dim = 768  # CLIP ViT-L/14 输出维度
        
        self.max_feats = 10      # 必须与攻击时一致
        self.max_tokens = 256
        self.max_atokens = 5
        self.use_video = True
        self.use_context = False
        self.suffix = ""
        self.dropout = 0.0
        self.use_adapter = True
        self.adapter_dim = 512
        self.n_ans = 2423        # MSVD 词表通常是 2423 或 1000，具体看 vocab.json
        self.freeze_last = False
        self.scratch = False 
        self.spatial_dim = 7
        self.dec_layers = 6
        self.enc_layers = 6
        self.freeze_lm = True   
        self.freeze_vn = True   
        self.freeze_mlm = True
        self.down_factor = 8     
        self.temporal_dim = 1    
        self.n_adapter_layers = 1 
        self.ft_ln = True
        self.ds_factor_attn = 8  
        self.ds_factor_ff = 8
        
        # 自动加载 vocab.json 以确定 n_ans (可选优化)
        vocab_path = "/data0/data/MSVD-QA/vocab.json"
        if os.path.exists(vocab_path):
            import json
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
                self.n_ans = len(vocab)
                print(f"[Config] Detected vocab size from file: {self.n_ans}")

args = EvalConfig()

# -----------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------
def load_qa_data(csv_path):
    """加载 QA 数据"""
    data_list = []
    
    # 尝试加载 vocab.json 来构建 a2id
    vocab_path = "/data0/data/MSVD-QA/vocab.json"
    a2id = {}
    if os.path.exists(vocab_path):
        import json
        with open(vocab_path, 'r') as f:
            a2id = json.load(f)
    else:
        print("Warning: vocab.json not found, constructing from CSV (might be inaccurate)")
        # Fallback: 从 CSV 构建 (不推荐)
        all_answers = set()
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_answers.add(row.get('answer','').strip())
        sorted_ans = sorted(list(all_answers))
        a2id = {a: i for i, a in enumerate(sorted_ans)}

    print(f"Loaded {len(a2id)} answers in vocabulary.")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get('video_id') or row.get('video') or row.get('id')
            if not vid: continue
            
            # 提取纯数字 ID (e.g. video100 -> 100)
            clean_id = re.sub(r"\D", "", str(vid))
            if not clean_id: continue
            
            q = row.get('question','').strip()
            a = row.get('answer','').strip()
            
            if a in a2id:
                data_list.append({
                    'vid_id': clean_id, 
                    'q': q,
                    'a_id': a2id[a]
                })

    return data_list, a2id

def init_answer_embeddings(model, tokenizer, a2id):
    """初始化答案 Embedding"""
    print(f"Initializing Answer Embeddings...")
    # 按照 id 顺序排列答案文本
    id2a = {v: k for k, v in a2id.items()}
    max_id = max(id2a.keys())
    
    # 必须保证 tensor 大小覆盖所有 ID
    # model.set_answer_embeddings 通常需要传入所有可能的 answer tokens
    # 这里我们只处理 top-N 答案
    
    # 构建 batch tokenization
    answers_text = [id2a.get(i, "") for i in range(max_id + 1)]
    
    # 批量 Tokenize
    # 注意显存，如果太多可以分批
    if len(answers_text) > 1000:
        print("Large vocabulary, processing in batches...")
    
    all_input_ids = []
    batch_size = 500
    for i in range(0, len(answers_text), batch_size):
        batch = answers_text[i:i+batch_size]
        toks = tokenizer(batch, add_special_tokens=False, max_length=args.max_atokens,
                        truncation=True, padding="max_length")["input_ids"]
        all_input_ids.append(torch.tensor(toks))
        
    aid2tokid = torch.cat(all_input_ids, dim=0).long()
    
    if hasattr(model, 'set_answer_embeddings'):
        model.set_answer_embeddings(aid2tokid.to(args.device), freeze_last=args.freeze_last)

# -----------------------------------------------------------
# 主函数
# -----------------------------------------------------------
def main():
    print(f"=== Starting Evaluation ===")
    
    # 1. 加载特征 (合并后的大文件)
    if not os.path.exists(args.adv_features_path):
        print(f"❌ Error: Feature file not found: {args.adv_features_path}")
        return
        
    print(f"Loading features from {args.adv_features_path}...")
    # 加载到 CPU 节省显存
    adv_features_dict = torch.load(args.adv_features_path, map_location='cpu')
    print(f"✅ Loaded {len(adv_features_dict)} video features.")
    
    # 2. 准备数据
    tokenizer = get_tokenizer(args)
    test_data, a2id = load_qa_data(args.qa_csv_path)
    print(f"Total Valid QA pairs: {len(test_data)}")
    
    # 3. 构建模型
    print("Building model...")
    model = build_model(args)
    
    # 加载权重
    print(f"Loading checkpoint: {args.load}")
    ckpt = torch.load(args.load, map_location='cpu')
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    
    # 处理 Projection 层 (768 -> 1536)
    # FrozenBiLM 官方 Checkpoint 通常包含这个权重
    # 如果特征是 768 (CLIP)，必须要有 Projection
    projector = nn.Linear(768, 1536)
    proj_key = 'deberta.embeddings.linear_video.weight'
    
    if proj_key in sd:
        # 确认形状
        if sd[proj_key].shape == (1536, 768):
            projector.weight.data = sd[proj_key]
            bias_key = proj_key.replace('.weight', '.bias')
            if bias_key in sd:
                projector.bias.data = sd[bias_key]
            print(f">>> Projection layer loaded successfully.")
        else:
            print(f"⚠️ Warning: Checkpoint projection shape {sd[proj_key].shape} mismatch with (1536, 768).")
    
    # 把 projector 挂载到模型上 (有些版本的 FrozenBiLM 内部会自动处理，我们手动挂载保险)
    model.v_project = projector
    
    # 加载主模型权重
    msg = model.load_state_dict(sd, strict=False)
    # print(f"Load status: {msg}") # 可选打印
    
    model.to(args.device)
    model.eval()
    
    # 初始化 Answer Embeddings
    init_answer_embeddings(model, tokenizer, a2id)

    # 4. 开始评估循环
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    total = 0
    missing = 0
    
    print("\nStarting Inference Loop...")
    
    for item in tqdm(test_data):
        vid_id = item['vid_id'] # 纯数字字符串 '100'
        question = item['q']
        target_id = item['a_id']
        
        # 查找特征
        # 尝试 int key 和 string key
        feat = None
        
        # 优先尝试 int key (torch.load 出来通常是 int)
        try:
            int_id = int(vid_id)
            if int_id in adv_features_dict:
                feat = adv_features_dict[int_id]
        except: pass
        
        # 其次尝试 string key
        if feat is None and vid_id in adv_features_dict:
            feat = adv_features_dict[vid_id]
            
        if feat is None:
            missing += 1
            if missing <= 5: print(f"[Debug] Missing feature for video {vid_id}")
            continue
            
        # 处理特征维度
        if not isinstance(feat, torch.Tensor): feat = torch.tensor(feat)
        feat = feat.float().to(args.device)
        
        # [10, 768] -> [1, 10, 768]
        if feat.dim() == 2: 
            feat = feat.unsqueeze(0)
            
        # 投影: [1, 10, 768] -> [1, 10, 1536]
        with torch.no_grad():
            projected_feat = model.v_project(feat)

        # 处理文本
        # FrozenBiLM 格式: "Question Answer: [MASK]."
        full_text = f"{question} Answer: [MASK]."
        encoded = tokenizer([full_text], add_special_tokens=True, max_length=args.max_tokens,
                            padding="longest", truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(args.device)
        attention_mask = encoded["attention_mask"].to(args.device)
        
        # 生成 Mask
        b, t, c = projected_feat.shape
        video_mask = get_mask(torch.tensor([t]), t).to(args.device)
        
        # 推理
        with torch.no_grad():
            output = model(
                video=None, # 我们手动传 video_embeds
                video_mask=video_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                video_embeds=projected_feat # 传入投影后的特征
            )
            
            logits = output['logits']
            
            # 定位 [MASK]
            mask_token_id = tokenizer.mask_token_id
            mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)
            
            if len(mask_indices[0]) > 0:
                # 考虑 offset (FrozenBiLM 拼接在前面)
                offset = t
                # 获取 mask 位置的 logits
                target_logits = logits[mask_indices[0], mask_indices[1] + offset, :]
                
                # 计算 Top-K
                # target_logits: [1, vocab_size]
                _, top10_indices = target_logits.topk(10, dim=-1)
                top10_ids = top10_indices[0].tolist()
                
                if target_id == top10_ids[0]: correct_1 += 1
                if target_id in top10_ids[:5]: correct_5 += 1
                if target_id in top10_ids[:10]: correct_10 += 1
                
                total += 1
            
    print(f"\n========================================")
    print(f"Evaluation Results (Pixel Attack):")
    print(f"Total Evaluated: {total}")
    print(f"Missing Features: {missing}")
    if total > 0:
        acc1 = correct_1 / total * 100
        acc5 = correct_5 / total * 100
        acc10 = correct_10 / total * 100
        print(f"Top-1 Accuracy:  {acc1:.2f}%")
        print(f"Top-5 Accuracy:  {acc5:.2f}%")
        print(f"Top-10 Accuracy: {acc10:.2f}%")
    else:
        print("No samples evaluated.")
    print(f"========================================")

if __name__ == "__main__":
    main()
