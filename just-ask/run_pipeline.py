import os
import sys
import glob
import pandas as pd
import subprocess
import json
import re
import csv
import torch
import numpy as np
import shutil
import time
import math
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. 配置区域
# -----------------------------------------------------------------------------
# [重要] 指定要使用的显卡 ID 列表
# 留空 [] 则自动检测所有可用显卡。
# 例如: GPU_IDS = [0, 1, 2, 3] 表示只用前4张卡
GPU_IDS = [1, 2, 3, 4] 

# 路径配置
JUSTASK_ROOT = "/data2/codefile/fjh/just-ask"
PROJ_ROOT = "/data2/codefile/fjh"

# 确保 Python 路径包含项目根目录
if PROJ_ROOT not in sys.path: sys.path.insert(0, PROJ_ROOT)
if JUSTASK_ROOT not in sys.path: sys.path.insert(0, JUSTASK_ROOT)

# 尝试导入 transformers (评测用)
try:
    from transformers import DistilBertTokenizer, BertTokenizer
    from model.multimodal_transformer import MMT_VideoQA
except ImportError:
    pass # 如果只是提取，这里报错暂时忽略

# --- 数据目录 ---
DATA_BASE_DIR = "/data0/data/msrvtt/"
VIDEOS_DIR = os.path.join(DATA_BASE_DIR, "videos")
RAW_FEATURES_DIR = os.path.join(DATA_BASE_DIR, "features_s3d_raw")
OUTPUT_PTH_PATH = os.path.join(DATA_BASE_DIR, "s3d.pth")

# --- 权重与模型路径 ---
S3D_WEIGHT_PATH = "/data0/pretrained/s3d/s3d_howto100m.pth"
S3D_DICT_PATH = "/data0/pretrained/s3d/s3d_dict.npy"
PRETRAIN_PATH = "/data0/pretrained/justask_checkpoints/ckpt_ft_msrvttqa.pth"
BERT_PATH = "/data0/pretrained/distilbert-base-uncased"
ANS_CSV = os.path.join(JUSTASK_ROOT, "MSRVTT-QA/test.csv")
VOCAB_PATH = os.path.join(JUSTASK_ROOT, "MSRVTT-QA/vocab.json")

# --- 参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FEATS = 20

# -----------------------------------------------------------------------------
# 2. 辅助函数
# -----------------------------------------------------------------------------
def get_available_gpus():
    """获取可用 GPU ID 列表"""
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
    """将列表平均切分为 num_chunks 份"""
    if num_chunks <= 0: return []
    k, m = divmod(len(data), num_chunks)
    return [data[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_chunks)]

def setup_environment():
    print(">>> [Step 0] Setting up environment...")
    if not os.path.exists(RAW_FEATURES_DIR):
        os.makedirs(RAW_FEATURES_DIR)
    
    # 修复包结构
    init_path = os.path.join(JUSTASK_ROOT, "extract", "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("# Auto-generated package init\n")

    # 创建软链接
    target_link = os.path.join(JUSTASK_ROOT, "s3d_dict.npy")
    if not os.path.exists(S3D_DICT_PATH):
        print(f"Error: Source file not found: {S3D_DICT_PATH}")
        sys.exit(1)
    if not os.path.exists(target_link):
        try:
            os.symlink(S3D_DICT_PATH, target_link)
        except OSError:
            shutil.copy(S3D_DICT_PATH, target_link)

# -----------------------------------------------------------------------------
# 3. 核心流程
# -----------------------------------------------------------------------------
def run_parallel_extraction():
    print("\n" + "="*60)
    print(">>> [Step 1] Preparing Tasks")
    print("="*60)
    
    # 1. 扫描视频
    video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(VIDEOS_DIR, ext)))
    video_files.sort()
    
    if not video_files:
        print(f"Error: No videos found in {VIDEOS_DIR}")
        sys.exit(1)

    # 2. 增量检查 (过滤掉已经提取过的)
    existing_feats = set(os.listdir(RAW_FEATURES_DIR))
    todos = []
    skipped = 0
    
    for v_path in video_files:
        fid = os.path.splitext(os.path.basename(v_path))[0]
        out_name = f"{fid}.npy"
        out_path = os.path.join(RAW_FEATURES_DIR, out_name)
        
        # 只要文件存在且大小正常(>1KB)，就跳过
        if out_name in existing_feats and os.path.getsize(out_path) > 1024:
            skipped += 1
            continue
        
        todos.append({'video_path': v_path, 'feature_path': out_path})
    
    print(f"Total videos: {len(video_files)}")
    print(f"Skipped (Done): {skipped}")
    print(f"Tasks remaining: {len(todos)}")
    
    if not todos:
        print("All tasks completed. Skipping directly to Merge.")
        return

    # 3. 分配任务给 GPU
    gpus = get_available_gpus()
    num_gpus = len(gpus)
    print(f"🚀 Launching extraction on {num_gpus} GPUs: {gpus}")
    
    chunks = split_list(todos, num_gpus)
    processes = []
    
    print("\n" + "="*60)
    print(">>> [Step 2] Launching Parallel Processes")
    print("="*60)

    for i, chunk in enumerate(chunks):
        if not chunk: continue # 显卡多于任务数的情况
        
        gpu_id = gpus[i]
        chunk_csv = os.path.join(DATA_BASE_DIR, f"temp_videos_gpu{gpu_id}.csv")
        pd.DataFrame(chunk).to_csv(chunk_csv, index=False)
        
        # 构造环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) # 关键：隔离显卡
        env["PYTHONPATH"] = f"{JUSTASK_ROOT}:{env.get('PYTHONPATH', '')}"
        # 确保能找到系统 ffmpeg
        if "/usr/bin" not in env.get("PATH", ""):
            env["PATH"] = f"/usr/bin:{env.get('PATH', '')}"

        # 构造命令
        # 注意：依然使用单线程解码 (num_decoding_thread=0)，因为我们在进程级别并行了
        cmd = [
            sys.executable, "-m", "extract.extract",
            "--csv", chunk_csv,
            "--model_path", S3D_WEIGHT_PATH,
            "--batch_size", "32", 
            "--num_decoding_thread", "0", 
            "--feature_dim", "1024",
            "--fps", "16"
        ]
        
        log_file = open(f"log_extract_gpu{gpu_id}.txt", "w")
        print(f"[GPU {gpu_id}] processing {len(chunk)} videos. Log: log_extract_gpu{gpu_id}.txt")
        
        # 异步启动
        p = subprocess.Popen(cmd, env=env, cwd=JUSTASK_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file))
    
    # 4. 等待结束
    print("\nWaiting for all processes to finish...")
    failed = False
    for p, f in processes:
        p.wait()
        f.close()
        if p.returncode != 0:
            failed = True
            print(f"Process failed with return code {p.returncode}. Check logs.")
    
    if failed:
        print("Warning: Some extraction processes failed.")
    else:
        print("All extraction processes finished successfully.")

def run_merge():
    print("\n" + "="*60)
    print(">>> [Step 3] Feature Merging")
    print("="*60)
    
    npy_files = glob.glob(os.path.join(RAW_FEATURES_DIR, "*.npy"))
    if not npy_files:
        print("No .npy files found to merge!")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{JUSTASK_ROOT}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "extract.merge_features",
        "--folder", RAW_FEATURES_DIR,
        "--output_path", OUTPUT_PTH_PATH,
        "--dataset", "msrvtt"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env, cwd=JUSTASK_ROOT)
        print(f"Features merged to: {OUTPUT_PTH_PATH}")
    except subprocess.CalledProcessError:
        print("Error in merging step.")

# --- 评测逻辑 (保持不变，因为只需要单卡跑一次) ---
def compute_a2v(vocab_path, bert_tokenizer, amax_words):
    with open(vocab_path, "r") as f: a2id = json.load(f)
    id2a = {v: k for k, v in a2id.items()}
    if not id2a: return a2id, torch.empty(0)
    answers = [id2a[i] for i in range(max(id2a.keys()) + 1)]
    enc = bert_tokenizer(answers, add_special_tokens=True, max_length=amax_words, padding="max_length", truncation=True, return_tensors="pt")
    return a2id, enc["input_ids"].long()

def load_qa_data(csv_path):
    data = []
    if not os.path.exists(csv_path): return data
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("video_id") and row.get("question") and row.get("answer"):
                data.append({'vid': re.sub(r"\.mp4$", "", row.get("video_id")), 'q': row.get("question").strip(), 'a': row.get("answer").strip()})
    return data

def run_evaluation():
    print("\n" + "="*60)
    print(">>> [Step 4] Evaluation")
    print("="*60)
    
    if not os.path.exists(OUTPUT_PTH_PATH):
        print(f"Error: Feature file {OUTPUT_PTH_PATH} does not exist.")
        return

    print("Loading features...")
    try:
        feats_dict = torch.load(OUTPUT_PTH_PATH, map_location='cpu')
    except Exception as e:
        print(f"Error loading features: {e}")
        return
    print(f"Loaded {len(feats_dict)} features.")

    # 简单检测一下 transformers 是否可用
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    except:
        try:
            tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
        except:
            print("Skipping evaluation due to missing transformers/model.")
            return

    a2id, a2v = compute_a2v(VOCAB_PATH, tokenizer, 10)
    
    # 评测只需要用一张卡，默认用第一张
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    a2v = a2v.to(device)

    model = MMT_VideoQA(
        feature_dim=1024, 
        word_dim=768, 
        N=2, 
        d_model=512,   # 确保这里是 512 (匹配 checkpoint)
        d_ff=2048,     # 确保这里是 2048
        h=8, 
        dropout=0.1, 
        T=MAX_FEATS, 
        Q=20, # QMAX_WORDS
        baseline=0
    ).to(DEVICE)
    if os.path.exists(PRETRAIN_PATH):
        sd = torch.load(PRETRAIN_PATH, map_location="cpu")
        if 'model_state_dict' in sd: sd = sd['model_state_dict']
        new_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
        model.load_state_dict(new_sd, strict=False)
    
    model.eval()
    with torch.no_grad(): model._compute_answer_embedding(a2v)
    
    qa_data = load_qa_data(ANS_CSV)
    correct_1, correct_10, total = 0, 0, 0
    
    for i in tqdm(range(0, len(qa_data), 128), desc="Evaluating"):
        batch = qa_data[i:i+128]
        if not batch: continue
        
        feats, qs, ans = [], [], []
        for item in batch:
            vid_str = item['vid']
            num_part = int(re.sub(r"\D", "", vid_str)) if re.sub(r"\D", "", vid_str) else None
            f = feats_dict.get(num_part) or feats_dict.get(vid_str)
            
            if f is not None:
                if not isinstance(f, torch.Tensor): f = torch.from_numpy(f)
                f = f.float()
                if f.dim() == 3: f = f.squeeze(0)
                if f.shape[0] < MAX_FEATS:
                    f = torch.cat([f, f[-1:].expand(MAX_FEATS-f.shape[0], -1)])
                else:
                    idx = torch.linspace(0, f.shape[0]-1, MAX_FEATS).long()
                    f = f[idx]
                feats.append(f)
                qs.append(item['q'])
                ans.append(a2id.get(item['a'], 0))
        
        if not feats: continue
        
        f_batch = torch.stack(feats).to(device)
        q_ids = tokenizer(qs, padding="max_length", truncation=True, max_length=20, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            logits = model(f_batch, q_ids, text_mask=(q_ids>0).float(), video_mask=torch.ones(len(qs), MAX_FEATS).to(device))
            preds = logits.argmax(1)
            correct_1 += (preds.cpu() == torch.tensor(ans)).sum().item()
            _, top10 = logits.topk(10, dim=1)
            correct_10 += (top10.cpu() == torch.tensor(ans).view(-1, 1)).any(dim=1).sum().item()
            total += len(qs)

    print("\n" + "="*60)
    print(f"Final Accuracy:")
    if total > 0:
        print(f"Top-1: {correct_1/total*100:.2f}%")
        print(f"Top-10: {correct_10/total*100:.2f}%")
    else:
        print("No data evaluated.")
    print("="*60)

if __name__ == "__main__":
    setup_environment()
    run_parallel_extraction()
    run_merge()
    run_evaluation()
