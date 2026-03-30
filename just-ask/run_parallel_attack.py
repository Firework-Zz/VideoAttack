#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多GPU并行对抗攻击主控脚本
功能：分配视频任务到多个GPU，收集对抗特征，最终评测
"""

import os
import sys
import glob
import pandas as pd
import subprocess
import json
import re
import csv
import torch
import shutil
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. 配置区域
# -----------------------------------------------------------------------------
GPU_IDS = [1, 3, 5]  # 指定要使用的GPU

# 路径配置
JUSTASK_ROOT = "/data2/codefile/fjh/just-ask"
PROJ_ROOT = "/data2/codefile/fjh"

if PROJ_ROOT not in sys.path: sys.path.insert(0, PROJ_ROOT)
if JUSTASK_ROOT not in sys.path: sys.path.insert(0, JUSTASK_ROOT)

# 数据目录
DATA_BASE_DIR = "/data0/data/msrvtt/"
VIDEOS_DIR = os.path.join(DATA_BASE_DIR, "videos")
RESULTS_BASE_DIR = "/data2/codefile/fjh/data/"
OUT_DIR = os.path.join(RESULTS_BASE_DIR, "pixel_attack_no_norm")
os.makedirs(OUT_DIR, exist_ok=True)

# 输出路径
RAW_ADV_FEATURES_DIR = os.path.join(OUT_DIR, "features_raw")  # 每个视频的对抗特征
OUTPUT_PTH_PATH = os.path.join(OUT_DIR, "s3d_adv_features_no_norm.pth")  # 合并后的特征文件
STATS_JSON_PATH = os.path.join(OUT_DIR, "attack_stats.json")  # 攻击统计

os.makedirs(RAW_ADV_FEATURES_DIR, exist_ok=True)

# QA数据路径
ANS_CSV = os.path.join(JUSTASK_ROOT, "MSRVTT-QA/test.csv")
VOCAB_PATH = os.path.join(JUSTASK_ROOT, "MSRVTT-QA/vocab.json")

# 预训练权重
PRETRAIN_PATH = "/data0/pretrained/justask_checkpoints/ckpt_ft_msrvttqa.pth"
S3D_WEIGHT_PATH = "/data0/pretrained/s3d/s3d_howto100m.pth"
BERT_PATH = "/data0/pretrained/distilbert-base-uncased"

# 攻击参数
NUM_KEYFRAMES = 20
PIXEL_EPS = 8.0 / 255.0
PIXEL_ALPHA = 2.0 / 255.0
PIXEL_STEPS = 10
MAX_FEATS = 20

# -----------------------------------------------------------------------------
# 2. 辅助函数
# -----------------------------------------------------------------------------
def get_available_gpus():
    if GPU_IDS:
        return GPU_IDS
    try:
        count = torch.cuda.device_count()
        if count == 0:
            print("Error: No GPUs detected!")
            sys.exit(1)
        return list(range(count))
    except:
        print("Error checking GPUs. Please set GPU_IDS manually.")
        sys.exit(1)

def split_list(data, num_chunks):
    if num_chunks <= 0: return []
    k, m = divmod(len(data), num_chunks)
    return [data[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_chunks)]

def load_qa_data(csv_path):
    """加载QA数据，返回 vid -> [qa_items] 的映射"""
    vid2qa = {}
    if not os.path.exists(csv_path): 
        print(f"Warning: QA file not found: {csv_path}")
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
            
            # 视频ID处理
            base = re.sub(r"\.mp4$", "", vid, flags=re.IGNORECASE)
            item = {'q': q, 'a': a, 'type': t}
            vid2qa.setdefault(base, []).append(item)
            
            # 同时存储纯数字版本
            num = re.sub(r"\D", "", base)
            if num and num != base:
                vid2qa.setdefault(num, []).append(item)
    
    return vid2qa

# -----------------------------------------------------------------------------
# 3. 核心流程
# -----------------------------------------------------------------------------
def prepare_tasks():
    """准备攻击任务列表"""
    print("\n" + "="*60)
    print(">>> [Step 1] Preparing Attack Tasks")
    print("="*60)
    
    # 1. 扫描所有视频
    video_files = glob.glob(os.path.join(VIDEOS_DIR, "*.mp4"))
    video_files.sort()
    
    if not video_files:
        print(f"Error: No videos found in {VIDEOS_DIR}")
        sys.exit(1)
    
    # 2. 加载QA数据
    vid2qa = load_qa_data(ANS_CSV)
    print(f"Loaded QA for {len(vid2qa)} videos")
    
    # 3. 增量检查（跳过已处理的视频）
    existing_feats = set(os.listdir(RAW_ADV_FEATURES_DIR))
    todos = []
    skipped = 0
    no_qa = 0
    
    for v_path in video_files:
        fname = os.path.basename(v_path)
        fid = os.path.splitext(fname)[0]
        
        # 提取视频ID（例如 video1234 -> 1234）
        vid_str = fid.split("video")[-1] if "video" in fid else fid
        
        # 检查是否有QA数据
        if vid_str not in vid2qa:
            no_qa += 1
            continue
        
        # 检查是否已处理
        out_name = f"{vid_str}.pth"
        if out_name in existing_feats:
            skipped += 1
            continue
        
        todos.append({
            'video_path': v_path,
            'video_id': vid_str,
            'num_qa': len(vid2qa[vid_str])
        })
    
    print(f"Total videos: {len(video_files)}")
    print(f"No QA data: {no_qa}")
    print(f"Skipped (Done): {skipped}")
    print(f"Tasks remaining: {len(todos)}")
    
    return todos

def run_parallel_attack(tasks):
    """并行执行攻击任务"""
    if not tasks:
        print("No tasks to process!")
        return
    
    print("\n" + "="*60)
    print(">>> [Step 2] Launching Parallel Attack Processes")
    print("="*60)
    
    gpus = get_available_gpus()
    num_gpus = len(gpus)
    print(f"🚀 Using {num_gpus} GPUs: {gpus}")
    
    # 分配任务
    chunks = split_list(tasks, num_gpus)
    processes = []
    
    for i, chunk in enumerate(chunks):
        if not chunk: continue
        
        gpu_id = gpus[i]
        
        # 保存任务列表到临时文件
        task_file = os.path.join(OUT_DIR, f"tasks_gpu{gpu_id}.json")
        with open(task_file, "w") as f:
            json.dump(chunk, f)
        
        # 环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{JUSTASK_ROOT}:{PROJ_ROOT}:{env.get('PYTHONPATH', '')}"
        
        # 命令
        cmd = [
            sys.executable, 
            "/data2/codefile/fjh/just-ask/attack_extractor.py",  # 下面会创建这个文件
            "--task_file", task_file,
            "--output_dir", RAW_ADV_FEATURES_DIR,
            "--gpu_id", str(gpu_id)
        ]
        
        log_file = open(os.path.join(OUT_DIR, f"log_attack_gpu{gpu_id}.txt"), "w")
        print(f"[GPU {gpu_id}] Processing {len(chunk)} videos. Log: log_attack_gpu{gpu_id}.txt")
        
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file))
    
    # 等待所有进程完成
    print("\nWaiting for all attack processes to finish...")
    failed = False
    for p, f in processes:
        p.wait()
        f.close()
        if p.returncode != 0:
            failed = True
            print(f"⚠️  Process failed with return code {p.returncode}")
    
    if failed:
        print("⚠️  Some processes failed. Check logs.")
    else:
        print("✅ All processes finished successfully!")

def merge_features():
    """合并所有对抗特征"""
    print("\n" + "="*60)
    print(">>> [Step 3] Merging Adversarial Features")
    print("="*60)
    
    pth_files = glob.glob(os.path.join(RAW_ADV_FEATURES_DIR, "*.pth"))
    if not pth_files:
        print("No .pth files found to merge!")
        return
    
    print(f"Found {len(pth_files)} feature files")
    
    merged_dict = {}
    for pth_path in tqdm(pth_files, desc="Merging"):
        try:
            feat = torch.load(pth_path, map_location='cpu')
            vid_id = os.path.splitext(os.path.basename(pth_path))[0]
            merged_dict[vid_id] = feat
        except Exception as e:
            print(f"Error loading {pth_path}: {e}")
    
    print(f"Saving merged features to {OUTPUT_PTH_PATH}...")
    torch.save(merged_dict, OUTPUT_PTH_PATH)
    print(f"✅ Merged {len(merged_dict)} features")

def collect_stats():
    """收集攻击统计数据"""
    print("\n" + "="*60)
    print(">>> [Step 4] Collecting Statistics")
    print("="*60)
    
    stats_files = glob.glob(os.path.join(OUT_DIR, "stats_gpu*.json"))
    if not stats_files:
        print("No stats files found")
        return
    
    all_stats = []
    for sf in stats_files:
        with open(sf, "r") as f:
            all_stats.extend(json.load(f))
    
    # 计算总体准确率
    total = len(all_stats)
    if total == 0:
        print("No statistics available")
        return
    
    clean_acc = sum(x['clean_hit'] for x in all_stats) / total * 100
    adv_acc = sum(x['adv_hit'] for x in all_stats) / total * 100
    
    # 按类型统计
    type_stats = {}
    for stat in all_stats:
        t = stat['type']
        if t not in type_stats:
            type_stats[t] = {'total': 0, 'clean': 0, 'adv': 0}
        type_stats[t]['total'] += 1
        type_stats[t]['clean'] += stat['clean_hit']
        type_stats[t]['adv'] += stat['adv_hit']
    
    print("\n" + "="*60)
    print("Overall Statistics:")
    print(f"Total QA pairs: {total}")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"Attack Success (Drop): {(clean_acc - adv_acc):.2f}%")
    print("="*60)
    
    if type_stats:
        print("\nPer-Type Statistics:")
        for t, stats in type_stats.items():
            c_acc = stats['clean'] / stats['total'] * 100
            a_acc = stats['adv'] / stats['total'] * 100
            print(f"  {t}: Clean={c_acc:.2f}%, Adv={a_acc:.2f}%, Drop={c_acc-a_acc:.2f}%")
    
    # 保存详细统计
    summary = {
        'total': total,
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'attack_drop': clean_acc - adv_acc,
        'type_statistics': type_stats,
        'all_stats': all_stats
    }
    
    with open(STATS_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Statistics saved to {STATS_JSON_PATH}")

def run_evaluation():
    """使用合并后的对抗特征进行评测"""
    print("\n" + "="*60)
    print(">>> [Step 5] Final Evaluation")
    print("="*60)
    
    if not os.path.exists(OUTPUT_PTH_PATH):
        print(f"Error: Feature file {OUTPUT_PTH_PATH} does not exist.")
        return
    
    try:
        from transformers import DistilBertTokenizer, BertTokenizer
        from model.multimodal_transformer import MMT_VideoQA
    except ImportError as e:
        print(f"Skipping evaluation due to import error: {e}")
        return
    
    print("Loading adversarial features...")
    feats_dict = torch.load(OUTPUT_PTH_PATH, map_location='cpu')
    print(f"Loaded {len(feats_dict)} adversarial features")
    
    # 加载tokenizer
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    except:
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    
    # 加载答案词表
    with open(VOCAB_PATH, "r") as f:
        a2id = json.load(f)
    id2a = {v: k for k, v in a2id.items()}
    answers = [id2a[i] for i in range(max(id2a.keys()) + 1)]
    enc = tokenizer(answers, add_special_tokens=True, max_length=10,
                   padding="max_length", truncation=True, return_tensors="pt")
    a2v = enc["input_ids"].long()
    
    # 加载模型
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    a2v = a2v.to(device)
    
    model = MMT_VideoQA(
        feature_dim=1024, word_dim=768, N=2, d_model=512, d_ff=2048,
        h=8, dropout=0.1, T=MAX_FEATS, Q=20, baseline=0
    ).to(device)
    
    if os.path.exists(PRETRAIN_PATH):
        sd = torch.load(PRETRAIN_PATH, map_location="cpu")
        if 'model_state_dict' in sd: sd = sd['model_state_dict']
        new_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
    
    model.eval()
    with torch.no_grad():
        model._compute_answer_embedding(a2v)
    
    # 加载QA数据
    qa_data = []
    with open(ANS_CSV, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            vid = row.get("video_id") or row.get("video")
            if not vid: continue
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not q or not a: continue
            vid_str = re.sub(r"\.mp4$", "", vid, flags=re.IGNORECASE)
            vid_str = vid_str.split("video")[-1] if "video" in vid_str else vid_str
            qa_data.append({'vid': vid_str, 'q': q, 'a': a})
    
    # 评测
    correct_1, correct_10, total = 0, 0, 0
    
    for i in tqdm(range(0, len(qa_data), 128), desc="Evaluating"):
        batch = qa_data[i:i+128]
        if not batch: continue
        
        feats, qs, ans = [], [], []
        for item in batch:
            vid_str = item['vid']
            f = feats_dict.get(vid_str)
            
            if f is None:
                continue
            
            if not isinstance(f, torch.Tensor):
                f = torch.from_numpy(f)
            f = f.float()
            
            # 确保维度正确
            if f.dim() == 3:
                f = f.squeeze(0)
            
            # 填充/采样到MAX_FEATS
            if f.shape[0] < MAX_FEATS:
                f = torch.cat([f, f[-1:].expand(MAX_FEATS - f.shape[0], -1)])
            elif f.shape[0] > MAX_FEATS:
                idx = torch.linspace(0, f.shape[0]-1, MAX_FEATS).long()
                f = f[idx]
            
            feats.append(f)
            qs.append(item['q'])
            ans.append(a2id.get(item['a'], 0))
        
        if not feats: continue
        
        f_batch = torch.stack(feats).to(device)
        q_ids = tokenizer(qs, padding="max_length", truncation=True, 
                         max_length=20, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            logits = model(f_batch, q_ids, 
                          text_mask=(q_ids > 0).float(),
                          video_mask=torch.ones(len(qs), MAX_FEATS).to(device))
            preds = logits.argmax(1)
            correct_1 += (preds.cpu() == torch.tensor(ans)).sum().item()
            
            _, top10 = logits.topk(10, dim=1)
            correct_10 += (top10.cpu() == torch.tensor(ans).view(-1, 1)).any(dim=1).sum().item()
            total += len(qs)
    
    print("\n" + "="*60)
    print("Final Evaluation Results:")
    if total > 0:
        print(f"Top-1 Accuracy: {correct_1/total*100:.2f}%")
        print(f"Top-10 Accuracy: {correct_10/total*100:.2f}%")
    else:
        print("No data evaluated.")
    print("="*60)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("Multi-GPU Parallel Adversarial Attack Pipeline")
    print("="*60)
    
    # Step 1: 准备任务
    tasks = prepare_tasks()
    
    # Step 2: 并行攻击
    run_parallel_attack(tasks)
    
    # Step 3: 合并特征
    merge_features()
    
    # Step 4: 收集统计
    collect_stats()
    
    # Step 5: 最终评测
    run_evaluation()
    
    print("\n" + "="*60)
    print("✅ Pipeline Complete!")
    print("="*60)

