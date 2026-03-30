"""
extract_msvd.py
修复：只提取 CLS Token，输出维度 [T, 1408]，解决 361856 维度错误。
"""

import sys
import os
import glob
import torch
import numpy as np
import math
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# ================= 0. 环境与路径 =================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from lavis.models.eva_vit import create_eva_vit_g
except ImportError:
    try:
        from lavis.models.eva_vit import create_eva_vit_g
    except:
        print("Error: Cannot import 'create_eva_vit_g'.")
        sys.exit(1)

# ================= 1. 配置参数 =================
FRAMES_ROOT = "/data0/data/msrvtt/frames_attack"
LOCAL_WEIGHT_PATH = "/data0/pretrained/MA-LMM/pretrained/eva_vit_g.pth"
# 建议输出到一个新文件夹，避免和之前错误的文件混淆
OUTPUT_DIR = "/data2/codefile/fjh/data/msvd_malmm_to_frozen" 

NUM_FRAMES = 10
BATCH_SIZE = 16
TARGET_GPUS = [0, 5, 7]

# ================= 2. 数据集 (保持不变) =================
def get_eva_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

class VideoFrameDataset(Dataset):
    def __init__(self, video_dirs, num_frames=10, transform=None):
        self.video_dirs = video_dirs
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        video_name = os.path.basename(video_dir)
        images = sorted(glob.glob(os.path.join(video_dir, "*.jpg")) + 
                        glob.glob(os.path.join(video_dir, "*.png")))
        if len(images) == 0:
            return video_name, torch.zeros(self.num_frames, 3, 224, 224)
        indices = np.linspace(0, len(images) - 1, self.num_frames, dtype=int)
        sampled_imgs = []
        for i in indices:
            try:
                img = Image.open(images[i]).convert("RGB")
                if self.transform: img = self.transform(img)
                sampled_imgs.append(img)
            except:
                sampled_imgs.append(torch.zeros(3, 224, 224))
        return video_name, torch.stack(sampled_imgs)

# ================= 3. 模型加载 (保持不变) =================
def load_model(device):
    model = create_eva_vit_g(img_size=224, drop_path_rate=0, use_checkpoint=False, precision="fp16")
    checkpoint = torch.load(LOCAL_WEIGHT_PATH, map_location="cpu")
    state_dict = checkpoint.get('model', checkpoint)
    new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device).half()
    model.eval()
    return model

# ================= 4. Worker 进程 (核心修复) =================
def worker_process(gpu_id, video_list, progress_idx):
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        model = load_model(device)
        dataset = VideoFrameDataset(video_list, num_frames=NUM_FRAMES, transform=get_eva_transform())
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        iterator = tqdm(dataloader, desc=f"GPU {gpu_id}", position=progress_idx) if progress_idx == 0 else dataloader

        with torch.no_grad():
            for batch_names, batch_pixels in iterator:
                B, T, C, H, W = batch_pixels.shape
                input_tensor = batch_pixels.view(B * T, C, H, W).to(device, dtype=torch.float16)
                
                # 1. 获取所有 Token 输出: [B*T, 257, 1408]
                full_features = model(input_tensor)
                
                # ==================== [核心修复] ====================
                # 不要只取 CLS ! 不要写 full_features[:, 0, :]
                # 我们需要保留全部空间特征: [B*T, 257, 1408]
                # ====================================================
                
                # 2. 恢复 Batch 和 Time 维度: [B, T, 257, 1408]
                # 注意：这里 view 的最后一维是 -1 (自动计算为 1408)，中间插入 257
                # 由于 full_features 已经是 [B*T, 257, 1408]，我们需要先 view 成 [B, T, 257, 1408]
                final_output = full_features.view(B, T, 257, -1)
                
                for i, vid_name in enumerate(batch_names):
                    save_path = os.path.join(OUTPUT_DIR, f"{vid_name}.pt")
                    # 保存完整的 4D Tensor: [T, 257, 1408]
                    torch.save(final_output[i].float().cpu(), save_path)
        
        print(f"[GPU {gpu_id}] Done.")
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")


# ================= 5. 主控逻辑 =================
def main():
    if not os.path.exists(FRAMES_ROOT):
        print("Frames root not found.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Scanning videos...")
    all_video_dirs = sorted([d for d in glob.glob(os.path.join(FRAMES_ROOT, "*")) if os.path.isdir(d)])
    total_videos = len(all_video_dirs)
    print(f"Total videos: {total_videos}")
    
    num_gpus = len(TARGET_GPUS)
    chunk_size = math.ceil(total_videos / num_gpus)
    splits = [all_video_dirs[i:i + chunk_size] for i in range(0, total_videos, chunk_size)]
    while len(splits) < num_gpus: splits.append([])

    print(f"Launching {num_gpus} processes...")
    mp.set_start_method('spawn', force=True)
    processes = []
    for i in range(num_gpus):
        if len(splits[i]) == 0: continue
        p = mp.Process(target=worker_process, args=(TARGET_GPUS[i], splits[i], i))
        p.start()
        processes.append(p)
    for p in processes: p.join()
    print("All tasks finished.")

if __name__ == "__main__":
    main()
