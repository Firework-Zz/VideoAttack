#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单GPU对抗攻击提取器
每个进程独立运行，对分配的视频执行攻击并提取对抗特征
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# 路径配置
JUSTASK_ROOT = "/data2/codefile/fjh/just-ask"
PROJ_ROOT = "/data2/codefile/fjh"
if PROJ_ROOT not in sys.path: sys.path.insert(0, PROJ_ROOT)
if JUSTASK_ROOT not in sys.path: sys.path.insert(0, JUSTASK_ROOT)

try:
    from utils.frame_sampling import sample_frames
    from model.multimodal_transformer import MMT_VideoQA
    from transformers import DistilBertTokenizer, BertTokenizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# 参数配置
FPS = 16
MAX_FEATS = 20
QMAX_WORDS = 20
AMAX_WORDS = 10

# 攻击参数
NUM_KEYFRAMES = 20
PIXEL_EPS = 8.0 / 255.0
PIXEL_ALPHA = 2.0 / 255.0
PIXEL_STEPS = 10

# 权重路径
S3D_WEIGHT_PATH = "/data0/pretrained/s3d/s3d_howto100m.pth"
PRETRAIN_PATH = "/data0/pretrained/justask_checkpoints/ckpt_ft_msrvttqa.pth"
BERT_PATH = "/data0/pretrained/distilbert-base-uncased"
VOCAB_PATH = os.path.join(JUSTASK_ROOT, "MSRVTT-QA/vocab.json")
ANS_CSV = os.path.join(JUSTASK_ROOT, "MSRVTT-QA/test.csv")

# 预处理（无Normalize，保持[0,1]）
preprocess_base = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()  # 输出 [0, 1]
])

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def get_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """生成mask矩阵"""
    rng = torch.arange(max_len, device=lengths.device).expand(lengths.size(0), max_len)
    return rng < lengths.unsqueeze(1)

feature_blobs = []
def hook_feature(module, input, output):
    """Hook函数，用于提取S3D特征"""
    feature_blobs.append(output)

def build_backbone(device):
    """构建S3D特征提取器"""
    from model.s3dg import S3D
    backbone = S3D(dict_path=S3D_WEIGHT_PATH, num_classes=400)
    
    if os.path.exists(S3D_WEIGHT_PATH):
        print(f"Loading S3D weights: {S3D_WEIGHT_PATH}")
        state_dict = torch.load(S3D_WEIGHT_PATH, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        new_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'fc.' not in k}
        backbone.load_state_dict(new_dict, strict=False)
    
    # 注册hook
    try:
        backbone.mixed_5c.register_forward_hook(hook_feature)
    except AttributeError:
        backbone.base.mixed_5c.register_forward_hook(hook_feature)
    
    for param in backbone.parameters():
        param.requires_grad = False
    
    return backbone.to(device).eval()

def load_qa_data(csv_path):
    """加载QA数据"""
    import csv
    import re
    
    vid2qa = {}
    if not os.path.exists(csv_path):
        return vid2qa
    
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            vid = row.get("video_id") or row.get("video")
            if not vid: continue
            
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            t = row.get("type", "default")
            
            if not q or not a: continue
            
            base = re.sub(r"\.mp4$", "", vid, flags=re.IGNORECASE)
            base = base.split("video")[-1] if "video" in base else base
            
            item = {'q': q, 'a': a, 'type': t}
            vid2qa.setdefault(base, []).append(item)
            
            # 同时存储纯数字版本
            num = re.sub(r"\D", "", base)
            if num and num != base:
                vid2qa.setdefault(num, []).append(item)
    
    return vid2qa

def compute_a2v(vocab_path, bert_tokenizer, amax_words):
    """计算答案词表的embedding"""
    with open(vocab_path, "r") as f:
        a2id = json.load(f)
    
    id2a = {v: k for k, v in a2id.items()}
    answers = [id2a[i] for i in range(max(id2a.keys()) + 1)]
    
    enc = bert_tokenizer(answers, add_special_tokens=True, max_length=amax_words,
                        padding="max_length", truncation=True, return_tensors="pt")
    
    return a2id, id2a, enc["input_ids"].long()

def encode_questions(tokenizer, questions, max_len, device):
    """编码问题文本"""
    enc = tokenizer(questions, padding="max_length", truncation=True,
                   max_length=max_len, return_tensors="pt")
    return enc.input_ids.to(device)

# -----------------------------------------------------------------------------
# 关键帧聚类
# -----------------------------------------------------------------------------
def cluster_frames_by_cosine(frames_ts, backbone, device, k=NUM_KEYFRAMES):
    """
    使用余弦相似度对帧进行聚类
    返回: 每帧所属的cluster ID (长度为T的tensor)
    """
    T = frames_ts.shape[0]
    if T <= k:
        return torch.arange(T, device=device)
    
    batch_size = 32
    feats_list = []
    
    # 提取所有帧的特征
    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch = frames_ts[i:i+batch_size].to(device)
            
            # 处理奇数帧
            real_bs = batch.shape[0]
            is_odd = (real_bs % 2 != 0)
            if is_odd:
                batch = torch.cat([batch, batch[-1:]], dim=0)
            
            # 转换为S3D输入格式: (B, C, T, H, W) -> (1, C, T, H, W)
            video_input = batch.permute(1, 0, 2, 3).unsqueeze(0)
            
            global feature_blobs
            feature_blobs = []
            _ = backbone(video_input)
            
            if not feature_blobs:
                f = torch.zeros(real_bs, 1024).to(device)
            else:
                f_raw = feature_blobs[0]
                # f_raw: (1, 1024, T', H', W')
                if f_raw.dim() == 5:
                    f = f_raw.mean(dim=[-2, -1])  # (1, 1024, T')
                    f = f.squeeze(0).permute(1, 0)  # (T', 1024)
                    
                    # 确保维度匹配
                    if f.shape[0] != batch.shape[0]:
                        f = f.unsqueeze(0).permute(0, 2, 1)
                        f = F.interpolate(f, size=batch.shape[0], mode='nearest')
                        f = f.permute(0, 2, 1).squeeze(0)
                else:
                    f = f_raw
            
            if is_odd:
                f = f[:-1]
            feats_list.append(f)
    
    all_feats = torch.cat(feats_list, dim=0)  # (T, 1024)
    all_feats_norm = F.normalize(all_feats, p=2, dim=1)
    
    # K-means聚类（基于余弦相似度）
    # 随机初始化k个中心
    indices = torch.randperm(T)[:k]
    centroids = all_feats_norm[indices]
    
    labels = None
    for _ in range(10):  # 迭代10次
        # 计算相似度矩阵
        sim_matrix = torch.matmul(all_feats_norm, centroids.t())  # (T, k)
        labels = sim_matrix.argmax(dim=1)  # (T,)
        
        # 更新中心
        new_centroids = []
        for i in range(k):
            mask = (labels == i)
            if mask.sum() > 0:
                cluster_mean = all_feats_norm[mask].mean(dim=0)
                new_centroids.append(F.normalize(cluster_mean, dim=0))
            else:
                # 如果某个cluster为空，随机选一个点
                new_centroids.append(all_feats_norm[torch.randint(0, T, (1,))].squeeze(0))
        centroids = torch.stack(new_centroids)
    
    return labels

# -----------------------------------------------------------------------------
# 对抗攻击
# -----------------------------------------------------------------------------
def run_cosine_keyframe_attack(
    frames_ts, qa_items, a2id, tokenizer,
    model, backbone, device, config
):
    """
    执行基于关键帧聚类的对抗攻击
    
    Args:
        frames_ts: (T, 3, 224, 224) 的原始帧
        qa_items: QA列表
        a2id: 答案到ID的映射
        tokenizer: BERT tokenizer
        model: MMT模型
        backbone: S3D骨干网络
        device: 设备
        config: 攻击配置
    
    Returns:
        adv_feat: 对抗特征 (MAX_FEATS, 1024)
        stats: 攻击统计信息
    """
    model.eval()
    backbone.eval()
    
    # 准备QA数据
    questions_text = [x['q'] for x in qa_items]
    answers_text = [x['a'] for x in qa_items]
    types_text = [x['type'] for x in qa_items]
    
    questions_ids = encode_questions(tokenizer, questions_text, QMAX_WORDS, device)
    answer_ids = torch.tensor([a2id.get(a, 0) for a in answers_text], dtype=torch.long, device=device)
    Q = len(questions_text)
    
    original_images = frames_ts.to(device)  # (T, 3, 224, 224)
    
    # Step 1: 关键帧聚类
    mapping = cluster_frames_by_cosine(frames_ts, backbone, device, k=config['num_keyframes'])
    actual_k = mapping.max().item() + 1
    
    # Step 2: 初始化扰动（只对关键帧）
    delta_key = torch.zeros((actual_k, 3, 224, 224), device=device)
    delta_key.requires_grad = True
    
    # 准备mask
    text_mask = (questions_ids > 0).float()
    video_len = torch.full((Q,), MAX_FEATS, dtype=torch.long, device=device)
    video_mask = get_mask(video_len, MAX_FEATS)
    
    # Step 3: PGD攻击
    for step in range(config['steps']):
        # 将关键帧扰动映射到所有帧
        delta_full = delta_key[mapping]  # (T, 3, 224, 224)
        adv_images = torch.clamp(original_images + delta_full, 0.0, 1.0)
        
        # 处理奇数帧
        curr_frames = adv_images
        if curr_frames.shape[0] % 2 != 0:
            curr_frames = torch.cat([curr_frames, curr_frames[-1:]], dim=0)
        
        # 转换为S3D输入格式
        inp = curr_frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, T, 224, 224)
        
        # 提取特征
        feature_blobs.clear()
        _ = backbone(inp)
        
        if not feature_blobs:
            raise RuntimeError("Hook failed to capture features")
        
        f_raw = feature_blobs[0]
        if f_raw.dim() == 5:
            feats = f_raw.mean(dim=[-2, -1])  # (1, 1024, T')
        else:
            feats = f_raw
        
        feats = feats.permute(0, 2, 1).squeeze(0)  # (T', 1024)
        
        # 采样/填充到MAX_FEATS
        K_feat = feats.shape[0]
        if K_feat >= MAX_FEATS:
            idx = torch.linspace(0, K_feat - 1, MAX_FEATS).long().to(device)
            v = feats.index_select(0, idx)
        else:
            v = torch.cat([feats, feats[-1:].expand(MAX_FEATS - K_feat, feats.shape[1])], dim=0)
        
        # 复制给所有问题
        video_feats_q = v.unsqueeze(0).expand(Q, -1, -1).contiguous()
        
        # 前向传播
        logits = model(video_feats_q, questions_ids, text_mask=text_mask, video_mask=video_mask)
        loss = F.cross_entropy(logits, answer_ids)
        
        # 反向传播
        model.zero_grad()
        backbone.zero_grad()
        if delta_key.grad is not None:
            delta_key.grad.zero_()
        loss.backward()
        
        # 更新扰动
        with torch.no_grad():
            delta_key.data = delta_key.data + config['alpha'] * delta_key.grad.sign()
            delta_key.data = torch.clamp(delta_key.data, -config['eps'], config['eps'])
    
    # Step 4: 提取最终对抗特征并评估
    with torch.no_grad():
        # 生成最终对抗样本
        final_delta = delta_key[mapping]
        final_adv_images = torch.clamp(original_images + final_delta, 0.0, 1.0)
        
        def extract_features(imgs):
            """从图像提取特征"""
            curr = imgs
            if curr.shape[0] % 2 != 0:
                curr = torch.cat([curr, curr[-1:]], dim=0)
            
            inp = curr.permute(1, 0, 2, 3).unsqueeze(0)
            
            feature_blobs.clear()
            _ = backbone(inp)
            
            if not feature_blobs:
                return torch.zeros(1, MAX_FEATS, 1024).to(device)
            
            f = feature_blobs[0]
            if f.dim() == 5:
                f = f.mean(dim=[-2, -1])
            f = f.permute(0, 2, 1).squeeze(0)
            
            kf = f.shape[0]
            if kf >= MAX_FEATS:
                idx = torch.linspace(0, kf - 1, MAX_FEATS).long().to(device)
                f_out = f.index_select(0, idx)
            else:
                f_out = torch.cat([f, f[-1:].expand(MAX_FEATS - kf, 1024)], dim=0)
            
            return f_out.unsqueeze(0)  # (1, MAX_FEATS, 1024)
        
        # 提取对抗特征
        feat_adv = extract_features(final_adv_images)
        logits_adv = model(feat_adv.expand(Q, -1, -1), questions_ids,
                          text_mask=text_mask, video_mask=video_mask)
        _, preds_adv = logits_adv.topk(10, dim=1)
        
        # 提取干净特征（用于对比）
        feat_clean = extract_features(original_images)
        logits_clean = model(feat_clean.expand(Q, -1, -1), questions_ids,
                            text_mask=text_mask, video_mask=video_mask)
        _, preds_clean = logits_clean.topk(10, dim=1)
        
        # 统计
        stats = []
        for i in range(Q):
            ans = answer_ids[i]
            clean_hit = (preds_clean[i, 0] == ans).item()
            adv_hit = (preds_adv[i, 0] == ans).item()
            stats.append({
                'type': types_text[i],
                'clean_hit': clean_hit,
                'adv_hit': adv_hit
            })
    
    return feat_adv.squeeze(0).cpu(), stats  # (MAX_FEATS, 1024), list

# -----------------------------------------------------------------------------
# Main处理函数
# -----------------------------------------------------------------------------
def process_tasks(args):
    """处理分配的任务列表"""
    device = torch.device(f"cuda:0")  # CUDA_VISIBLE_DEVICES已设置
    print(f"Using device: {device}")
    
    # 1. 加载任务列表
    with open(args.task_file, "r") as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks from {args.task_file}")
    
    # 2. 加载模型和tokenizer
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    except:
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    
    a2id, id2a, a2v = compute_a2v(VOCAB_PATH, tokenizer, AMAX_WORDS)
    a2v = a2v.to(device)
    
    # 加载MMT模型
    model = MMT_VideoQA(
        feature_dim=1024, word_dim=768, N=2, d_model=512, d_ff=2048,
        h=8, dropout=0.1, T=MAX_FEATS, Q=QMAX_WORDS, baseline=0
    ).to(device)
    
    if os.path.exists(PRETRAIN_PATH):
        state = torch.load(PRETRAIN_PATH, map_location="cpu")
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        new_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        model.load_state_dict(new_sd, strict=False)
    
    with torch.no_grad():
        model._compute_answer_embedding(a2v)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # 加载S3D backbone
    backbone = build_backbone(device)
    
    # 3. 加载QA数据
    vid2qa = load_qa_data(ANS_CSV)
    print(f"Loaded QA for {len(vid2qa)} videos")
    
    # 4. 攻击配置
    attack_config = {
        'num_keyframes': NUM_KEYFRAMES,
        'eps': PIXEL_EPS,
        'alpha': PIXEL_ALPHA,
        'steps': PIXEL_STEPS
    }
    
    # 5. 处理每个视频
    all_stats = []
    success_count = 0
    
    for task in tqdm(tasks, desc=f"GPU {args.gpu_id}"):
        video_path = task['video_path']
        video_id = task['video_id']
        
        try:
            # 5.1 加载视频帧
            frames = sample_frames(video_path, fps=FPS)
            if not frames:
                print(f"Warning: No frames extracted from {video_path}")
                continue
            
            # 确保至少2帧（S3D要求）
            if len(frames) < 2:
                frames = frames * 2
            
            # 预处理
            frames_ts = torch.stack([preprocess_base(Image.fromarray(fr).convert("RGB")) 
                                    for fr in frames])
            
            # 5.2 获取QA数据
            qa_items = vid2qa.get(video_id, [])
            if not qa_items:
                print(f"Warning: No QA data for {video_id}")
                continue
            
            # 5.3 执行攻击
            adv_feat, stats = run_cosine_keyframe_attack(
                frames_ts, qa_items, a2id, tokenizer,
                model, backbone, device, attack_config
            )
            
            # 5.4 保存对抗特征
            out_path = os.path.join(args.output_dir, f"{video_id}.pth")
            torch.save(adv_feat, out_path)
            
            # 5.5 收集统计
            all_stats.extend(stats)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 6. 保存统计数据
    stats_path = os.path.join(os.path.dirname(args.output_dir), f"stats_gpu{args.gpu_id}.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n✅ GPU {args.gpu_id} finished!")
    print(f"   Processed: {success_count}/{len(tasks)} videos")
    print(f"   Stats saved to: {stats_path}")

# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", type=str, required=True, help="Task list JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID")
    args = parser.parse_args()
    
    process_tasks(args)

