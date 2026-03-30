# eval_frozenbilm.py
import sys
import os
import csv
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# === [Patch] 修复 Numpy 版本兼容性问题 ===
if not hasattr(np, 'int'):
    np.int = int

# 1. 环境设置
sys.path.append(os.getcwd())
try:
    from model import build_model, get_tokenizer
    from util.misc import get_mask
except ImportError:
    print("[ERROR] 请确保脚本在 FrozenBiLM 根目录下运行")
    sys.exit(1)

# -----------------------------------------------------------
# 配置参数 (必须与攻击时保持一致)
# -----------------------------------------------------------
class EvalConfig:
    def __init__(self):
        # === [路径配置] 请确认这些路径正确 ===
        self.output_dir = "/data0/data/msrvtt/malmm_to_frozenbilm" # 你的对抗特征存放目录
        self.qa_csv_path = "/data0/pretrained/FrozenBilm/data/test.csv"
        self.load = "/data0/pretrained/FrozenBilm/checkpoints/frozenbilm_msrvtt.pth"
        self.model_name = "/data0/pretrained/deberta-v2-xlarge" 
        os.environ["TRANSFORMERS_CACHE"] = "/data0/pretrained/"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # === [关键配置] 必须是 768 (适配 CLIP-Base) ===
        self.features_dim = 768 
        
        self.max_feats = 10
        self.max_tokens = 32
        self.max_atokens = 5
        self.use_video = True
        self.use_context = False
        self.suffix = ""
        self.dropout = 0.0 # 测试时关闭 dropout
        self.use_adapter = True
        self.adapter_dim = 512
        self.n_ans = 1500
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

args = EvalConfig()

# -----------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------
def load_qa_data(csv_path):
    """加载 QA 数据并生成答案映射"""
    data_list = []
    all_answers = set()
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get('video_id') or row.get('video')
            if not vid: continue
            
            # 提取纯数字 ID (例如 video7010 -> 7010)
            vid_key = vid.replace('video','').replace('.mp4','').split('.')[0]
            
            q = row.get('question','').strip()
            a = row.get('answer','').strip()
            all_answers.add(a)
            
            data_list.append({
                'vid_key': vid_key, # '7010'
                'q': q,
                'a': a
            })

    sorted_ans = sorted(list(all_answers))
    if len(sorted_ans) > args.n_ans:
        sorted_ans = sorted_ans[:args.n_ans]
    a2id = {a: i for i, a in enumerate(sorted_ans)}
    return data_list, a2id

def init_answer_embeddings(model, tokenizer, a2id):
    """初始化答案 Embedding"""
    print(f"Initializing Answer Embeddings ({len(a2id)})...")
    vocab_size = max(args.n_ans, len(a2id))
    aid2tokid = torch.zeros(vocab_size, args.max_atokens).long()
    for a, aid in a2id.items():
        if aid >= vocab_size: continue
        tok = torch.tensor(
            tokenizer(a, add_special_tokens=False, max_length=args.max_atokens,
                      truncation=True, padding="max_length")["input_ids"],
            dtype=torch.long,
        )
        aid2tokid[aid] = tok
    
    if hasattr(model, 'set_answer_embeddings'):
        model.set_answer_embeddings(aid2tokid.to(args.device), freeze_last=args.freeze_last)

# -----------------------------------------------------------
# 主函数
# -----------------------------------------------------------
def main():
    print(f"=== Starting Evaluation ===")
    
    # 1. 准备数据
    tokenizer = get_tokenizer(args)
    test_data, a2id = load_qa_data(args.qa_csv_path)
    print(f"Total QA pairs: {len(test_data)}")
    
    # 2. 构建模型
    model = build_model(args)
    
    # 3. 手动加载投影层 (768 -> 1536)
    print("Loading v_project projection layer (768->1536)...")
    projector = nn.Linear(768, 1536)
    
    # 加载权重
    ckpt = torch.load(args.load, map_location='cpu')
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    
    # 查找投影层权重 (CLIP-Base 对应的名字通常是这个)
    proj_key = 'deberta.embeddings.linear_video.weight'
    
    # 智能查找
    if proj_key not in sd:
        for k, v in sd.items():
            if v.ndim == 2 and v.shape == (1536, 768):
                proj_key = k
                break
    
    if proj_key in sd:
        projector.weight.data = sd[proj_key]
        bias_key = proj_key.replace('.weight', '.bias')
        if bias_key in sd:
            projector.bias.data = sd[bias_key]
        print(f">>> Projection layer loaded from: {proj_key}")
    else:
        raise RuntimeError("FATAL: 无法在权重中找到 [1536, 768] 的投影层！请检查 checkpoint。")
            
    model.v_project = projector
    
    # 4. 加载模型其他权重
    model.load_state_dict(sd, strict=False)
    
    model.to(args.device)
    model.eval()
    
    # 初始化 Answer Embeddings
    init_answer_embeddings(model, tokenizer, a2id)

        # 5. 开始评估循环
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    total = 0
    missing = 0
    debug_limit = 5 
    
    print("\nStarting Inference Loop...")
    
    for item in tqdm(test_data):
        vid_key = item['vid_key']
        question = item['q']
        answer = item['a']
        
        # 目标答案 ID
        if answer not in a2id:
            continue
        target_id = a2id[answer]
        
        # 文件名匹配
        name_candidates = [
            f"video{vid_key}_adv.pt",
            f"{vid_key}_adv.pt"
        ]
        
        adv_path = None
        for name in name_candidates:
            p = os.path.join(args.output_dir, name)
            if os.path.exists(p):
                adv_path = p
                break
        
        if adv_path is None:
            missing += 1
            if missing <= debug_limit:
                print(f"[DEBUG] File Not Found. Tried: {name_candidates}")
            continue
            
        # 加载特征
        try:
            video_feats = torch.load(adv_path, map_location=args.device)
            if video_feats.dim() == 2: 
                video_feats = video_feats.unsqueeze(0)
            video_feats = video_feats.float()
        except Exception as e:
            print(f"Error loading {adv_path}: {e}")
            continue

        # 处理文本
        encoded = tokenizer([question], add_special_tokens=True, max_length=args.max_tokens,
                            padding="longest", truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(args.device)
        attention_mask = encoded["attention_mask"].to(args.device)
        
        # 生成 Mask
        b, t, c = video_feats.shape
        video_mask = get_mask(torch.tensor([t]), t).to(args.device)
        
        # 推理
        with torch.no_grad():
            projected_feats = model.v_project(video_feats)
            output = model(
                video=None,
                video_mask=video_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                video_embeds=projected_feats
            )
            
            logits = output['logits']
            
            # 处理 logits 维度
            delay = args.max_feats if args.use_video else 0
            if logits.dim() == 3:
                mask_indices = (input_ids == tokenizer.mask_token_id)
                if mask_indices.any():
                    target_logits = logits[:, delay : input_ids.size(1) + delay][mask_indices]
                else:
                    target_logits = logits[:, 0, :]
            else:
                target_logits = logits
            
            # === [修改] 计算 Top-1, Top-5, Top-10 ===
            # 取出分数最高的 10 个 ID
            # target_logits shape: [1, vocab_size]
            _, top10_indices = target_logits.topk(10, dim=-1) # [1, 10]
            top10_ids = top10_indices[0].tolist() # 转成 list: [id1, id2, ..., id10]
            
            # 统计 Top-1
            if target_id == top10_ids[0]:
                correct_1 += 1
            
            # 统计 Top-5
            if target_id in top10_ids[:5]:
                correct_5 += 1
                
            # 统计 Top-10
            if target_id in top10_ids[:10]:
                correct_10 += 1
                
            total += 1
            
    print(f"\n========================================")
    print(f"Evaluation Results:")
    print(f"Total Evaluated: {total}")
    print(f"Missing Files:   {missing}")
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

