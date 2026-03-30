import sys
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.multiprocessing as mp

# ==================== 0. 路径与环境配置 ====================
current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
lavis_dir = os.path.dirname(models_dir)
project_root = os.path.dirname(lavis_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Error: 请安装 decord")
    sys.exit(1)

from lavis.models.eva_vit import create_eva_vit_g

# 常量配置
VIDEOS_DIR = "/data0/data/msrvtt/videos" 
OUT_DIR = "/data2/codefile/fjh/data/feats_pgd_answer"
EVA_CKPT_PATH = "/data0/pretrained/MA-LMM/pretrained/eva_vit_g.pth"
NUM_FRAMES = 10

# 【核心修改】：指定你想要使用的物理显卡 ID 列表
TARGET_GPUS = [0, 1, 2, 3, 5, 7]

# ==================== 处理器与逻辑函数 ====================
class ManualProcessor:
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    def __call__(self, item):
        return self.transform(item)

def get_video_frames(video_path):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        indices = np.linspace(0, len(vr) - 1, NUM_FRAMES).astype(int)
        batch_data = vr.get_batch(indices)
        frames = batch_data.asnumpy() if hasattr(batch_data, "asnumpy") else batch_data.numpy()
        return [Image.fromarray(frm) for frm in frames]
    except Exception:
        return None

# ==================== 多进程 Worker 函数 ====================
def worker(gpu_id, video_list, progress_pos):
    """
    gpu_id: 实际的物理显卡 ID
    video_list: 该显卡负责的视频路径列表
    progress_pos: tqdm 进度条显示的位置（防止多行进度条重叠）
    """
    # 1. 强制绑定物理显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:0") # 经过上面的设置，这里 cuda:0 就指向了物理 gpu_id
    
    # 2. 每个进程独立加载模型
    try:
        model = create_eva_vit_g(img_size=224, precision="fp16").to(device)
        checkpoint = torch.load(EVA_CKPT_PATH, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval().half()
        
        vis_processor = ManualProcessor(image_size=224)
        
        # 3. 开始处理
        desc = f"GPU {gpu_id}"
        for video_path in tqdm(video_list, desc=desc, position=progress_pos, leave=True):
            video_name = os.path.basename(video_path).split('.')[0]
            save_path = os.path.join(OUT_DIR, f"{video_name}.pt")
            
            if os.path.exists(save_path):
                continue

            pil_images = get_video_frames(video_path)
            if pil_images is None:
                continue

            image_input = torch.stack([vis_processor(img) for img in pil_images]).to(device).half()

            with torch.no_grad():
                features = model.forward_features(image_input)
                torch.save(features.cpu(), save_path)
    except Exception as e:
        print(f"[GPU {gpu_id}] 发生严重错误: {e}")

# ==================== 主控函数 ====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. 扫描并筛选视频 (video7010-video9999)
    all_video_files = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    video_files = []
    for f in all_video_files:
        video_name = os.path.basename(f).split('.')[0]
        try:
            video_idx = int(video_name.replace('video', ''))
            if 7010 <= video_idx <= 9999:
                video_files.append(f)
        except ValueError:
            continue
    
    num_total = len(video_files)
    num_workers = len(TARGET_GPUS)
    print(f"准备使用显卡: {TARGET_GPUS}，共分配 {num_total} 个任务。")
    
    # 2. 切分任务列表
    video_splits = np.array_split(video_files, num_workers)
    
    # 3. 启动多进程
    # 显卡多进程必须使用 spawn 模式
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for i in range(num_workers):
        # 参数说明: 
        # TARGET_GPUS[i]: 实际物理卡号
        # video_splits[i]: 分配的视频列表
        # i: 进度条在屏幕上的行号
        p = mp.Process(target=worker, args=(TARGET_GPUS[i], video_splits[i].tolist(), i))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print(f"\n[Done] 所有显卡任务已完成！")

if __name__ == "__main__":
    main()
