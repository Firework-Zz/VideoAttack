#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSVD 对抗攻击全流程脚本 (Fix: Missing Import 'F')
修复内容：
1. 补全 import torch.nn.functional as F
2. 保持 MAX_FEATS = 20 以匹配权重维度
3. 包含所有之前的路径修复和自动 QA 加载逻辑
"""

import os
import sys
import glob
import json
import re
import csv
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F  # <---【关键修复】补上了这一行
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

# =============================================================================
# 1. 全局配置
# =============================================================================
# GPU 设置
GPU_IDS = [2, 6]

# 路径设置
JUSTASK_ROOT = "/data2/codefile/fjh/just-ask"
PROJ_ROOT = "/data2/codefile/fjh"
DATA_BASE_DIR = "/data0/data/msvd/" 
MSVD_QA_ROOT = "/data0/data/MSVD-QA"

# 添加 Python 路径
if PROJ_ROOT not in sys.path: sys.path.insert(0, PROJ_ROOT)
if JUSTASK_ROOT not in sys.path: sys.path.insert(0, JUSTASK_ROOT)

# 具体文件路径
VIDEOS_DIR = os.path.join(DATA_BASE_DIR, "videos")
ANS_CSV = os.path.join(MSVD_QA_ROOT, "test.csv") 
VOCAB_PATH = os.path.join(MSVD_QA_ROOT, "vocab.json")
PRETRAIN_PATH = "/data0/pretrained/justask_checkpoints/ckpt_ft_msvdqa.pth"
S3D_WEIGHT_PATH = "/data0/pretrained/s3d/s3d_howto100m.pth"
BERT_PATH = "/data0/pretrained/distilbert-base-uncased"

# 输出设置
RESULTS_BASE_DIR = "/data2/codefile/fjh/data_msvd/" 
OUT_DIR = os.path.join(RESULTS_BASE_DIR, "pixel_attack_no_norm")
RAW_ADV_FEATURES_DIR = os.path.join(OUT_DIR, "features_raw")
OUTPUT_PTH_PATH = os.path.join(OUT_DIR, "s3d_adv_features_msvd.pth")

# 攻击参数
MAX_FEATS = 20  # 必须是 20，配合 QMAX_WORDS=20，总长 40 才能匹配权重
QMAX_WORDS = 20
AMAX_WORDS = 10
PIXEL_EPS = 8.0 / 255.0
PIXEL_ALPHA = 2.0 / 255.0
PIXEL_STEPS = 10

# 确保输出目录存在
os.makedirs(RAW_ADV_FEATURES_DIR, exist_ok=True)

# =============================================================================
# 2. 模型定义
# =============================================================================

try:
    from model.multimodal_transformer import MMT_VideoQA
    from transformers import DistilBertTokenizer, BertTokenizer
    try:
        from model.s3dg import S3D
    except ImportError:
        from model.s3d import S3D
except ImportError as e:
    print(f"❌ [Critical Error] Import failed: {e}")
    sys.exit(1)

feature_blobs = []
def hook_feature(module, input, output):
    feature_blobs.append(output)

class S3DWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()
        try:
            self.backbone = S3D(dict_path=S3D_WEIGHT_PATH, num_classes=400)
        except TypeError:
            self.backbone = S3D(S3D_WEIGHT_PATH, num_classes=400)
            
        self.backbone.to(device).eval()
        for p in self.backbone.parameters(): p.requires_grad = False
        
        target_layer = None
        if hasattr(self.backbone, 'mixed_5c'): target_layer = self.backbone.mixed_5c
        elif hasattr(self.backbone, 'base'): target_layer = self.backbone.base.mixed_5c
        
        if target_layer: target_layer.register_forward_hook(hook_feature)

    def forward(self, x):
        x_norm = (x - 0.5) * 2.0
        global feature_blobs
        feature_blobs = []
        
        out = self.backbone(x_norm)
        
        feat = None
        if len(feature_blobs) > 0:
            feat = feature_blobs[-1]
        elif isinstance(out, dict):
            for k in ['mixed_5c', 'video_embedding', 'logits']:
                if k in out:
                    feat = out[k]
                    break
            if feat is None: feat = list(out.values())[0]
        else:
            feat = out
            
        if feat is not None:
            if feat.dim() == 5: feat = feat.mean(dim=[-2, -1])
            if feat.dim() == 2: feat = feat.unsqueeze(-1)
            
        return feat 

# =============================================================================
# 3. 数据处理函数
# =============================================================================

def load_all_qa_data(root_dir):
    vid2qa = {}
    csv_files = glob.glob(os.path.join(root_dir, "*.csv"))
    
    for csv_path in csv_files:
        try:
            f = open(csv_path, "r", encoding='utf-8-sig')
        except:
            f = open(csv_path, "r", encoding='utf-8')
        with f:
            rdr = csv.DictReader(f)
            id_keys = ['video_id', 'video', 'id']
            for row in rdr:
                vid = None
                for k in id_keys:
                    if row.get(k):
                        vid = row[k]
                        break
                if not vid: continue
                
                q = (row.get("question") or "").strip()
                a = (row.get("answer") or "").strip()
                t = row.get("type", "default")
                if not q or not a: continue
                
                clean_id = re.sub(r"\D", "", str(vid))
                if clean_id:
                    vid2qa.setdefault(clean_id, []).append({'q': q, 'a': a, 'type': t})
    return vid2qa

def get_video_frames(video_path, num_frames=32):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened(): return None
    
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= 0: return None

    if v_len <= num_frames:
        indices = np.arange(v_len)
    else:
        indices = np.linspace(0, v_len - 1, num_frames).astype(int)
        
    for i in range(v_len):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
        
    if len(frames) == 0: return None
    return np.array(frames) 

def compute_a2v(vocab_path, bert_tokenizer):
    with open(vocab_path, "r") as f: a2id = json.load(f)
    id2a = {v: k for k, v in a2id.items()}
    answers = [id2a[i] for i in range(max(id2a.keys()) + 1)]
    enc = bert_tokenizer(answers, add_special_tokens=True, max_length=AMAX_WORDS,
                        padding="max_length", truncation=True, return_tensors="pt")
    return a2id, id2a, enc["input_ids"].long()

def encode_questions(tokenizer, questions, device):
    enc = tokenizer(questions, padding="max_length", truncation=True,
                   max_length=QMAX_WORDS, return_tensors="pt")
    return enc.input_ids.to(device)

# =============================================================================
# 4. Worker 进程逻辑
# =============================================================================

def worker_process(gpu_id, task_list, result_queue=None):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[GPU {gpu_id}] Initializing on {device}...")
        
        vid2qa = load_all_qa_data(MSVD_QA_ROOT)
        
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
        except:
            tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
        a2id, id2a, a2v = compute_a2v(VOCAB_PATH, tokenizer)
        a2v = a2v.to(device)
        
        # 初始化 MMT (MAX_FEATS=20)
        mmt = MMT_VideoQA(feature_dim=1024, word_dim=768, N=2, d_model=512, d_ff=2048, 
                          h=8, dropout=0.1, T=MAX_FEATS, Q=QMAX_WORDS, baseline=0).to(device)
        if os.path.exists(PRETRAIN_PATH):
            sd = torch.load(PRETRAIN_PATH, map_location="cpu")
            if 'model_state_dict' in sd: sd = sd['model_state_dict']
            mmt.load_state_dict({k.replace('module.', ''): v for k,v in sd.items()}, strict=False)
        with torch.no_grad(): mmt._compute_answer_embedding(a2v)
        mmt.eval()
        
        s3d = S3DWrapper(device)
        
        count = 0
        pbar = tqdm(task_list, desc=f"GPU {gpu_id}", position=gpu_id)
        
        for task in pbar:
            vid_id = str(task['video_id'])
            vid_path = task['video_path']
            out_path = os.path.join(RAW_ADV_FEATURES_DIR, f"{vid_id}.pth")
            
            if os.path.exists(out_path): continue
            
            if vid_id not in vid2qa:
                clean = re.sub(r"\D", "", vid_id)
                if clean in vid2qa: vid_id = clean
                else: continue
            
            qa_pairs = vid2qa[vid_id]
            
            try:
                frames_np = get_video_frames(vid_path, num_frames=MAX_FEATS)
                if frames_np is None: continue
                
                frames_t = torch.from_numpy(frames_np).float() / 255.0
                frames_t = frames_t.permute(3, 0, 1, 2).unsqueeze(0)
                
                B, C, T, H, W = frames_t.shape
                if H != 224 or W != 224:
                    frames_reshaped = frames_t.squeeze(0).permute(1, 0, 2, 3) 
                    frames_resized = F.interpolate(frames_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
                    frames_t = frames_resized.permute(1, 0, 2, 3).unsqueeze(0)
                
                feat = run_attack_logic(frames_t, qa_pairs, a2id, tokenizer, mmt, s3d, device)
                
                torch.save(feat, out_path)
                count += 1
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error {vid_id}: {e}")
                continue
        
        print(f"[GPU {gpu_id}] Finished. Processed {count} videos.")
        
    except Exception as e:
        print(f"❌ [GPU {gpu_id}] Crashed: {e}")
        import traceback
        traceback.print_exc()

def run_attack_logic(frames_tensor, qa_items, a2id, tokenizer, mmt_model, s3d_wrapper, device):
    mmt_model.eval()
    qs = [x['q'] for x in qa_items]
    ans = [x['a'] for x in qa_items]
    
    q_ids = encode_questions(tokenizer, qs, device)
    a_ids = torch.tensor([a2id.get(a, 0) for a in ans], dtype=torch.long, device=device)
    Q = len(qs)
    
    video_clean = frames_tensor.to(device)
    delta = torch.zeros_like(video_clean).to(device)
    delta.requires_grad = True
    
    text_mask = (q_ids > 0).float()
    video_mask = torch.ones(Q, MAX_FEATS).to(device)
    
    for step in range(PIXEL_STEPS):
        adv_video = torch.clamp(video_clean + delta, 0.0, 1.0)
        s3d_feat = s3d_wrapper(adv_video)
        if s3d_feat is None: continue
        
        feat_t = s3d_feat.squeeze(0).permute(1, 0) 
        K = feat_t.shape[0]
        if K >= MAX_FEATS:
            idx = torch.linspace(0, K-1, MAX_FEATS).long().to(device)
            feat_final = feat_t.index_select(0, idx)
        else:
            feat_final = torch.cat([feat_t, feat_t[-1:].expand(MAX_FEATS-K, -1)])
            
        feat_batch = feat_final.unsqueeze(0).expand(Q, -1, -1)
        logits = mmt_model(feat_batch, q_ids, text_mask=text_mask, video_mask=video_mask)
        loss = F.cross_entropy(logits, a_ids)
        
        mmt_model.zero_grad()
        if delta.grad is not None: delta.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            delta.data = delta.data + PIXEL_ALPHA * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -PIXEL_EPS, PIXEL_EPS)

    with torch.no_grad():
        final_adv_video = torch.clamp(video_clean + delta, 0.0, 1.0)
        final_feat_raw = s3d_wrapper(final_adv_video)
        if final_feat_raw is not None:
            ft = final_feat_raw.squeeze(0).permute(1, 0)
            if ft.shape[0] >= MAX_FEATS:
                idx = torch.linspace(0, ft.shape[0]-1, MAX_FEATS).long().to(device)
                final_feat_out = ft.index_select(0, idx)
            else:
                final_feat_out = torch.cat([ft, ft[-1:].expand(MAX_FEATS-ft.shape[0], -1)])
        else:
            final_feat_out = torch.zeros(MAX_FEATS, 1024).cpu()

    return final_feat_out.cpu()

# =============================================================================
# 5. 主程序入口
# =============================================================================

def build_id_mapping():
    mapping = {} 
    map_files = [os.path.join(MSVD_QA_ROOT, "youtube_mapping.txt"), os.path.join(DATA_BASE_DIR, "youtube_mapping.txt")]
    found = next((f for f in map_files if os.path.exists(f)), None)
    if found:
        print(f"Loaded mapping from {found}")
        with open(found, 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p)>=2: mapping[p[0]] = re.sub(r"\D", "", p[1])
    return mapping

def prepare_tasks():
    video_files = glob.glob(os.path.join(VIDEOS_DIR, "*"))
    video_files = [f for f in video_files if f.endswith(('.avi', '.mp4', '.mkv'))]
    video_files.sort()
    
    name2id = build_id_mapping()
    # 预加载
    vid2qa = {}
    if os.path.exists(ANS_CSV):
        try: f = open(ANS_CSV, "r", encoding='utf-8-sig')
        except: f = open(ANS_CSV, "r", encoding='utf-8')
        with f:
            for r in csv.DictReader(f):
                v = r.get("video_id") or r.get("video")
                if v: vid2qa[re.sub(r"\D", "", v)] = 1

    todos = []
    existing = set(os.listdir(RAW_ADV_FEATURES_DIR))
    
    print(f"Scanning {len(video_files)} videos...")
    for v_path in video_files:
        base = os.path.splitext(os.path.basename(v_path))[0]
        tid = None
        if base in name2id: tid = name2id[base]
        elif base in vid2qa: tid = base
        else:
            clean = re.sub(r"\D", "", base)
            if clean in vid2qa: tid = clean
        
        # Fallback for mapping
        if not tid and base in name2id: tid = name2id[base]
        if not tid: tid = re.sub(r"\D", "", base)
            
        if tid:
            if f"{tid}.pth" not in existing:
                todos.append({'video_path': v_path, 'video_id': tid})
    
    print(f"Generated {len(todos)} tasks.")
    return todos

def split_list(data, n):
    k, m = divmod(len(data), n)
    return [data[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def main():
    mp.set_start_method('spawn', force=True)
    print(">>> All-in-One Attack Script Started")
    tasks = prepare_tasks()
    
    if not tasks:
        print("No tasks found.")
        return

    gpus = GPU_IDS
    chunks = split_list(tasks, len(gpus))
    
    processes = []
    for i, chunk in enumerate(chunks):
        if not chunk: continue
        gpu = gpus[i]
        p = mp.Process(target=worker_process, args=(gpu, chunk))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("✅ All processes finished.")

if __name__ == "__main__":
    main()
