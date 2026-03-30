#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrozenBiLM 像素级对抗攻击主控脚本 (MSVD 增强版)
功能：
1. 自动加载 youtube_mapping.txt 解决 ID 匹配问题
2. 支持断点续传 (会自动跳过 features_raw 下已存在的文件)
"""

import os
import sys
import glob
import subprocess
import json
import re
import csv
import torch
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. 配置区域
# -----------------------------------------------------------------------------
GPU_IDS = [0, 1, 2, 3]  # 使用 4 张卡

# 路径配置
FROZEN_ROOT = "/data2/codefile/fjh/FrozenBilm"
DATA_BASE_DIR = "/data0/data/msvd/" 
VIDEOS_DIR = os.path.join(DATA_BASE_DIR, "videos")
MSVD_QA_ROOT = "/data0/data/MSVD-QA"

# 关键：ID 映射文件路径 (用于把 youtube文件名 转换为 videoID)
MAPPING_FILE = os.path.join(MSVD_QA_ROOT, "youtube_mapping.txt")
# 备用路径 (以防文件在 dataset 根目录)
MAPPING_FILE_BACKUP = os.path.join(DATA_BASE_DIR, "youtube_mapping.txt")

# Checkpoint
CHECKPOINT_PATH = "/data0/pretrained/FrozenBilm/checkpoints/frozenbilm_msvd.pth" 
ANS_CSV = os.path.join(MSVD_QA_ROOT, "test.csv")

# 输出目录
RESULTS_BASE_DIR = "/data2/codefile/fjh/FrozenBilm/attack_results_msvd"
OUT_DIR = os.path.join(RESULTS_BASE_DIR, "clip_pixel_attack")
os.makedirs(OUT_DIR, exist_ok=True)

RAW_ADV_FEATURES_DIR = os.path.join(OUT_DIR, "features_raw")
os.makedirs(RAW_ADV_FEATURES_DIR, exist_ok=True)

OUTPUT_PTH_PATH = os.path.join(OUT_DIR, "frozenbilm_adv_clip_msvd.pth")

# 攻击参数
PIXEL_EPS = 8.0 / 255.0
PIXEL_ALPHA = 2.0 / 255.0
PIXEL_STEPS = 10
WORKER_SCRIPT_NAME = "frozenbilm_attack_worker.py"

# -----------------------------------------------------------------------------
# 2. 辅助函数
# -----------------------------------------------------------------------------
def split_list(data, num_chunks):
    if num_chunks <= 0: return []
    k, m = divmod(len(data), num_chunks)
    return [data[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_chunks)]

def load_id_mapping():
    """加载 youtube_mapping.txt: {filename_hash: video_id_num}"""
    mapping = {}
    target_file = None
    
    if os.path.exists(MAPPING_FILE): target_file = MAPPING_FILE
    elif os.path.exists(MAPPING_FILE_BACKUP): target_file = MAPPING_FILE_BACKUP
    
    if target_file:
        print(f"Loaded ID mapping from {target_file}")
        with open(target_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # 格式通常是: [original_name] [video_id]
                    # 例如: -4wsuPCjDBc_5_15 video1
                    orig = parts[0]
                    vid_str = parts[1] # video1
                    vid_num = re.sub(r"\D", "", vid_str) # 1
                    mapping[orig] = vid_num
    else:
        print("⚠️ Warning: youtube_mapping.txt not found. Matching might fail for MSVD.")
        
    return mapping

def load_qa_data(csv_path):
    vid2qa = {}
    if not os.path.exists(csv_path):
        print(f"Error: QA CSV not found at {csv_path}")
        return {}
    
    try: f = open(csv_path, "r", encoding='utf-8-sig')
    except: f = open(csv_path, "r", encoding='utf-8')

    with f:
        rdr = csv.DictReader(f)
        for row in rdr:
            vid = row.get("video_id") or row.get("video") or row.get("id")
            q = row.get("question")
            a = row.get("answer")
            
            if vid and q and a:
                clean_id = re.sub(r"\D", "", str(vid))
                if clean_id:
                    vid2qa.setdefault(clean_id, []).append({'q': q, 'a': a})
    return vid2qa

# -----------------------------------------------------------------------------
# 3. 任务准备
# -----------------------------------------------------------------------------
def prepare_tasks():
    print(">>> [Step 1] Preparing Tasks...")
    
    # 1. 扫描视频
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm']:
        video_files.extend(glob.glob(os.path.join(VIDEOS_DIR, ext)))
    video_files.sort()
    
    # 2. 加载 QA 和 映射
    vid2qa = load_qa_data(ANS_CSV)
    name2id = load_id_mapping()
    
    print(f"Total Video Files: {len(video_files)}")
    print(f"Total QA Entries: {len(vid2qa)}")
    
    tasks = []
    skipped = 0
    existing = set(os.listdir(RAW_ADV_FEATURES_DIR))
    
    for v_path in video_files:
        v_name = os.path.basename(v_path)
        base = os.path.splitext(v_name)[0]
        
        # --- ID 匹配逻辑 ---
        target_id = None
        
        # 1. 尝试通过 mapping 文件匹配
        if base in name2id:
            target_id = name2id[base]
        
        # 2. 如果没匹配上，尝试文件名本身就是 ID (例如 video100)
        if not target_id:
            raw_num = re.sub(r"\D", "", base)
            if raw_num and raw_num in vid2qa:
                target_id = raw_num
        
        # 3. 还没匹配上，跳过
        if not target_id:
            continue
            
        # 4. 检查 QA 是否存在
        if target_id not in vid2qa:
            continue
            
        # 5. 检查是否已完成 (断点续传)
        # Worker 保存的文件名为 {video_id}.pth
        if f"{target_id}.pth" in existing:
            skipped += 1
            continue
            
        tasks.append({
            'video_path': v_path,
            'video_id': target_id,
            'qa_pairs': vid2qa[target_id]
        })
        
    print(f"Skipped (Already done): {skipped}")
    print(f"Tasks to run: {len(tasks)}")
    
    if len(tasks) == 0 and skipped == 0:
        print("❌ Warning: No tasks generated. Please check paths and mapping file.")
        
    return tasks

# -----------------------------------------------------------------------------
# 4. 执行与合并
# -----------------------------------------------------------------------------
def run_parallel_attack(tasks):
    if not tasks: return
    
    print(">>> [Step 2] Launching Parallel Attack...")
    worker_script = os.path.join(FROZEN_ROOT, WORKER_SCRIPT_NAME)
    
    # 均匀分配
    chunks = split_list(tasks, len(GPU_IDS))
    processes = []
    
    for i, chunk in enumerate(chunks):
        if not chunk: continue
        gpu_id = GPU_IDS[i]
        
        task_file = os.path.join(OUT_DIR, f"tasks_gpu{gpu_id}.json")
        with open(task_file, "w") as f: json.dump(chunk, f)
            
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{FROZEN_ROOT}:{env.get('PYTHONPATH', '')}"
        
        cmd = [
            sys.executable, "-u", worker_script,
            "--task_file", task_file,
            "--output_dir", RAW_ADV_FEATURES_DIR,
            "--gpu_id", str(gpu_id),
            "--eps", str(PIXEL_EPS),
            "--alpha", str(PIXEL_ALPHA),
            "--steps", str(PIXEL_STEPS),
            "--checkpoint", CHECKPOINT_PATH
        ]
        
        log_path = os.path.join(OUT_DIR, f"log_gpu{gpu_id}.txt")
        log_file = open(log_path, "w")
        
        print(f"🚀 GPU {gpu_id}: Processing {len(chunk)} videos... (Log: {log_path})")
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file))
        
    for p, f in processes:
        p.wait()
        f.close()
    print("✅ Attack finished.")

def merge_features():
    print(">>> [Step 3] Merging Features...")
    pth_files = glob.glob(os.path.join(RAW_ADV_FEATURES_DIR, "*.pth"))
    if not pth_files: 
        print("No files to merge.")
        return

    merged = {}
    for f in tqdm(pth_files):
        v_id = os.path.splitext(os.path.basename(f))[0]
        try:
            feat = torch.load(f, map_location='cpu')
            if isinstance(feat, torch.Tensor):
                try: k = int(v_id)
                except: k = v_id
                merged[k] = feat
        except: pass
            
    torch.save(merged, OUTPUT_PTH_PATH)
    print(f"✅ Saved merged features to {OUTPUT_PTH_PATH} ({len(merged)} videos)")

if __name__ == "__main__":
    # 如果你想重跑，取消下面这行的注释来清空旧数据
    # import shutil; shutil.rmtree(RAW_ADV_FEATURES_DIR, ignore_errors=True); os.makedirs(RAW_ADV_FEATURES_DIR, exist_ok=True)
    
    tasks = prepare_tasks()
    run_parallel_attack(tasks)
    merge_features()
