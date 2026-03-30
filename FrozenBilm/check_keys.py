import torch
import cv2
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
from model.frozenbilm import FrozenBiLM
from transformers import DebertaV2Tokenizer

# ================= 配置 =================
# 随便找一个存在的视频路径
TEST_VIDEO = "/data0/data/msrvtt/videos/video0.mp4" 
# 假设 video0 的内容大概是生活场景，我们造两个句子
TEXT_TRUE = "a girl is talking to the camera in a kitchen" # 假设这是真的（或者从 csv 里找个真的）
TEXT_FALSE = "a fighter jet is flying in the sky"          # 这肯定是假的
# =======================================

# 配置路径
CHECKPOINT = "/data0/pretrained/FrozenBilm/checkpoints/frozenbilm_msrvtt.pth"
TOKENIZER_PATH = "/data0/pretrained/deberta-v2-xlarge/" # 必须用 DeBERTa 分词器

def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    # 简单取 10 帧
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, v_len-1, 10).astype(int)
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cap.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    return np.array(frames)

def main():
    print("1. Loading Model...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(TOKENIZER_PATH)
    # config 只要能跑就行，权重靠 load_state_dict
    config = {"pretrained_clip_name": "ViT-B/32", "vocab_size": 128100}
    model = FrozenBiLM(config)
    
    print("2. Loading Weights...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Load Msg: {msg}")
    
    model.cuda().eval()
    
    print("3. Preparing Data...")
    frames = get_frames(TEST_VIDEO)
    # Norm & Tensor
    frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    video = ((frames - mean) / std).unsqueeze(0).cuda() # [1, 10, 3, 224, 224]
    
    # Tokenize Text
    texts = [TEXT_TRUE, TEXT_FALSE]
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
    
    print("4. Running Inference...")
    with torch.no_grad():
        # 我们这里把同一个视频复制两份，分别对应两句话
        video_batch = video.repeat(2, 1, 1, 1, 1)
        out = model(video=video_batch, input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        scores = out['scores'] # [Score_True, Score_False]
        
    print("\n" + "="*40)
    print(f"视频路径: {TEST_VIDEO}")
    print(f"句子 A (真?): '{TEXT_TRUE}' \t 得分: {scores[0].item():.4f}")
    print(f"句子 B (假?): '{TEXT_FALSE}' \t 得分: {scores[1].item():.4f}")
    print("="*40)
    
    if scores[0] > scores[1]:
        print("✅ 测试通过！模型认为句子 A 更匹配。")
        print(f"   信心差距: {scores[0] - scores[1]:.4f}")
    else:
        print("❌ 测试失败！模型认为句子 B 更匹配（瞎猜）。")

if __name__ == "__main__":
    main()
