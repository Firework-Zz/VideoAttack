import sys
import os
import glob
import csv
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from transformers import LlamaTokenizer

# ==================== 0. 路径与环境配置 ====================
current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
lavis_dir = os.path.dirname(models_dir)
project_root = os.path.dirname(lavis_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lavis.models import load_model_and_preprocess

# ==================== 1. 全局配置参数 ====================
# 【攻击参数】
FEAT_EPS = 8 / 255
FEAT_ALPHA = 0.01
FEAT_STEPS = 10       
QA_BATCH_SIZE = 16     
QMAX_WORDS = 32

# 【新增：早停参数】
EARLY_STOP_PATIENCE = 5     
EARLY_STOP_THRESHOLD = 1e-4 

# 【数据路径】
CLEAN_FEATS_DIR = "/data2/codefile/fjh/data/feats_pgd_answer_clean"
OUT_DIR = "/data2/codefile/fjh/data/feats_pgd_feature" 
QA_CSV_PATH = "/data2/codefile/fjh/just-ask/MSRVTT-QA/test.csv"

# 【模型路径】
VICUNA_PATH = "/data0/pretrained/MA-LMM/llm/vicuna-7b/"
MALMM_CKPT_PATH = "/data0/pretrained/MA-LMM/saved_model/MSRVTT_qa/checkpoint_best.pth"

# 【显卡配置】
TARGET_GPUS = [0,1,2,3]

# ==================== 2. 工具函数 ====================
def load_qa_from_csv(csv_path: str):
    vid2qa = {}
    if not os.path.exists(csv_path): return {}
    with open(csv_path, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id") or row.get("video")
            if not vid: continue
            q, a = (row.get("question") or "").strip(), (row.get("answer") or "").strip()
            if not q or not a: continue
            vid_key = vid.replace("video", "").split(".")[0]
            if vid_key not in vid2qa: vid2qa[vid_key] = []
            vid2qa[vid_key].append({'question': q, 'answer': a})
    return vid2qa

# ==================== 3. 核心攻击逻辑 ====================

def run_attack_on_raw_feats(model, tokenizer, raw_feats_1408, qa_list, device):
    """
    MA-LMM 增强版攻击: QA Loss + Latent Disruption
    """
    model.eval()
    
    # 1. 准备 Clean 数据 (用于对比破坏程度)
    video_orig = raw_feats_1408.detach().to(device, dtype=torch.float16)
    
    with torch.no_grad():
        with autocast(enabled=True, dtype=torch.float16):
            img_embeds_clean = model.ln_vision(video_orig)
            img_atts_clean = torch.ones(img_embeds_clean.size()[:-1], dtype=torch.long).to(device)
            query_tokens_clean = model.query_tokens.expand(img_embeds_clean.shape[0], -1, -1)
            query_output_clean = model.Qformer.bert(
                query_embeds=query_tokens_clean,
                encoder_hidden_states=img_embeds_clean,
                encoder_attention_mask=img_atts_clean,
                return_dict=True,
            )
            inputs_llm_clean = model.llm_proj(query_output_clean.last_hidden_state)
            inputs_llm_clean = inputs_llm_clean.detach()

    # 2. 初始化 PGD
    # 建议放大一点 EPS，ViT 特征的方差通常比较大
    # 如果效果还不好，尝试把 FEAT_EPS 增大到 0.5 甚至 1.0 测试一下敏感度
    delta = torch.zeros_like(video_orig, requires_grad=True, dtype=torch.float32)
    delta.data.uniform_(-FEAT_EPS, FEAT_EPS)
    
    momentum_buffer = torch.zeros_like(video_orig, dtype=torch.float32)
    
    questions = [item['question'] for item in qa_list]
    answers = [item['answer'] for item in qa_list]

    for step in range(FEAT_STEPS):
        if delta.grad is not None:
            delta.grad.zero_()
            
        total_loss_val = 0.0
        
        # Batch Loop
        for i in range(0, len(questions), QA_BATCH_SIZE):
            batch_q = questions[i : i + QA_BATCH_SIZE]
            batch_a = answers[i : i + QA_BATCH_SIZE]
            current_bs = len(batch_q)
            
            # 构造输入
            prompts = [f"Question: {q} Answer:" for q in batch_q]
            full_texts = [f"Question: {q} Answer: {a}{tokenizer.eos_token}" for q, a in zip(batch_q, batch_a)]
            
            prompt_tokens = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True, max_length=QMAX_WORDS).to(device)
            full_tokens = tokenizer(full_texts, return_tensors="pt", padding="longest", truncation=True).to(device)
            
            # 叠加扰动
            adv_feats_1408 = (video_orig + delta).to(torch.float16)
            
            with autocast(enabled=True, dtype=torch.float16):
                # === Forward: Visual -> Q-Former -> Proj ===
                image_embeds = model.ln_vision(adv_feats_1408)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                
                query_output = model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                inputs_llm = model.llm_proj(query_output.last_hidden_state)
                
                # Expand features for QA batch
                # clean_ref (用于计算距离)
                clean_ref = inputs_llm_clean.mean(dim=0, keepdim=True).expand(current_bs, -1, -1)
                # current_adv
                video_feature_agg = inputs_llm.mean(dim=0, keepdim=True).expand(current_bs, -1, -1)
                
                # === Forward: LLM ===
                full_text_embeds = model.llm_model.get_input_embeddings()(full_tokens.input_ids)
                inputs_embeds = torch.cat([video_feature_agg, full_text_embeds], dim=1)
                
                attention_mask = torch.cat([
                    torch.ones(video_feature_agg.shape[:2], dtype=torch.long, device=device),
                    full_tokens.attention_mask
                ], dim=1)
                
                labels = full_tokens.input_ids.clone()
                labels[:, :prompt_tokens.input_ids.shape[1]] = -100
                labels[labels == tokenizer.pad_token_id] = -100
                final_labels = torch.cat([
                    torch.full((current_bs, video_feature_agg.shape[1]), -100, dtype=torch.long, device=device),
                    labels
                ], dim=1)
                
                outputs = model.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=final_labels,
                    return_dict=True
                )
                
                # === Loss Calculation ===
                # 1. QA Loss: 让模型预测变得糟糕 (Maximize CrossEntropy)
                loss_qa = outputs.loss
                
                # 2. Mid-Layer Disruption Loss: 让 Q-Former 输出乱套
                # 我们希望 Adv 和 Clean 的特征在余弦空间上距离尽可能远
                # CosineSim(A, B) -> 1 (Similar), -1 (Opposite)
                # 我们要 Maximize Loss。
                # 设置 Loss = -CosineSim。当 A,B 很像(1) -> Loss=-1 (小)。当 A,B 不像(0) -> Loss=0 (大)。
                # 这样梯度会推动 A 远离 B。
                loss_disrupt = -F.cosine_similarity(video_feature_agg, clean_ref, dim=-1).mean()
                
                # 3. 组合
                # 加大 disruption 的权重，因为这是直接梯度
                final_loss = loss_qa + 5.0 * loss_disrupt
                
                # 缩放 loss 避免梯度过小 (因为用了 FP16)
                weighted_loss = final_loss * (current_bs / len(questions))
            
            weighted_loss.backward()
            total_loss_val += final_loss.item()
        
        # print(f"  Step {step}: Loss {total_loss_val:.4f}")

        # === PGD Update ===
        with torch.no_grad():
            if delta.grad is None: continue
            
            grad = delta.grad
            grad_norm = torch.norm(grad, p=2)
            if grad_norm > 1e-8:
                grad = grad / grad_norm
            
            momentum_buffer = 0.9 * momentum_buffer + grad # 常用动量系数 0.9
            
            # 更新
            delta.data = delta.data + FEAT_ALPHA * momentum_buffer.sign()
            
            # 截断
            delta.data = torch.clamp(delta.data, -FEAT_EPS, FEAT_EPS)
            
            # 归零梯度
            delta.grad.zero_()

    final_adv_feats = (video_orig + delta).detach().cpu()
    return final_adv_feats

# ==================== 4. Worker 进程 ====================

def worker(gpu_id, feat_file_list, vid2qa, progress_pos):
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        print(f"[GPU {gpu_id}] Loading Model on {device}...", flush=True)
        
        model, _, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct_malmm", 
            model_type="vicuna7b", 
            is_eval=True, 
            device=device 
        )
        
        tokenizer = LlamaTokenizer.from_pretrained(VICUNA_PATH, use_fast=False)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        ckpt_malmm = torch.load(MALMM_CKPT_PATH, map_location="cpu")
        model.load_state_dict(ckpt_malmm, strict=False)
        
        print(f"[GPU {gpu_id}] Pruning Visual Encoder (Keeping Q-Former)...", flush=True)
        if hasattr(model, 'visual_encoder'): del model.visual_encoder
        import gc; gc.collect(); torch.cuda.empty_cache()
        
        model = model.to(device=device, dtype=torch.float16)
        model.eval()
        
        desc = f"GPU {gpu_id}"
        for feat_path in tqdm(feat_file_list, desc=desc, position=progress_pos, leave=True):
            fname = os.path.basename(feat_path)
            save_path = os.path.join(OUT_DIR, fname)
            
            if os.path.exists(save_path):
                continue
            
            video_idx_str = fname.replace('video', '').split('.')[0]
            qa_pairs = vid2qa.get(video_idx_str)
            if not qa_pairs:
                continue

            try:
                raw_feats_1408 = torch.load(feat_path, map_location='cpu') 
                adv_feats_1408 = run_attack_on_raw_feats(model, tokenizer, raw_feats_1408, qa_pairs, device)
                torch.save(adv_feats_1408, save_path)
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error attacking {fname}: {e}")
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error: {e}")
        import traceback
        traceback.print_exc()

# ==================== 5. 主控函数 ====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Loading QA data...")
    vid2qa = load_qa_from_csv(QA_CSV_PATH)
    
    print(f"Scanning clean features from: {CLEAN_FEATS_DIR}")
    all_feat_files = sorted(glob.glob(os.path.join(CLEAN_FEATS_DIR, "*.pt")))
    
    tasks = []
    skipped_count = 0 
    for f in all_feat_files:
        filename = os.path.basename(f)
        save_path = os.path.join(OUT_DIR, filename)
        if os.path.exists(save_path):
            skipped_count += 1
            continue 
        try:
            video_idx = int(filename.replace('video', '').split('.')[0])
            if 7010 <= video_idx <= 9999 and str(video_idx) in vid2qa:
                tasks.append(f)
        except ValueError:
            continue
    
    num_total = len(tasks)
    num_workers = len(TARGET_GPUS)
    print(f"准备使用显卡: {TARGET_GPUS}，共分配 {num_total} 个任务。")
    
    if num_total == 0:
        print("未找到任务。")
        return

    file_splits = np.array_split(tasks, num_workers)
    
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(TARGET_GPUS[i], file_splits[i].tolist(), vid2qa, i))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print(f"\n[Done] All tasks finished.")

if __name__ == "__main__":
    main()
