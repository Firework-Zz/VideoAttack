import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# ==================== 0. 路径修复 (强制把当前目录加入环境) ====================
current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
lavis_dir = os.path.dirname(models_dir)
project_root = os.path.dirname(lavis_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"[Init] Project Root: {project_root}")

# ==================== 1. 尝试导入关键库 ====================
print("[Init] 正在导入库...")
try:
    from decord import VideoReader, cpu
    print("   [OK] Decord imported.")
except ImportError:
    print("   [Error] Decord 导入失败！")
    sys.exit(1)

# 导入 EVA-VIT
try:
    from lavis.models.eva_vit import create_eva_vit_g
    print("   [OK] EVA-ViT imported from lavis.")
except ImportError:
    try:
        from eva_vit import create_eva_vit_g
        print("   [OK] EVA-ViT imported from local.")
    except ImportError:
        print("   [Error] 找不到 eva_vit.py，无法加载模型结构！")
        sys.exit(1)

# ==================== 2. 配置 ====================
VIDEOS_DIR = "/data0/data/msrvtt/videos" 
OUT_DIR = "/data2/codefile/fjh/data/feats_pgd_answer"
EVA_CKPT_PATH = "/data0/pretrained/MA-LMM/pretrained/eva_vit_g.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 3. 定义处理器 (与之前一致) ====================
class ManualProcessor:
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
    def __call__(self, item):
        return self.transform(item)

# ==================== 4. 核心调试流程 ====================
def test_one_video():
    print(f"\n=== 开始单视频提取测试 ===")
    
    # A. 获取第一个视频
    import glob
    video_files = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    if not video_files:
        print("[Error] 没找到 mp4 文件")
        return
    
    target_video = video_files[0] # 取第一个
    video_name = os.path.basename(target_video).split('.')[0]
    save_path = os.path.join(OUT_DIR, f"{video_name}.pt")
    
    print(f"[Step 1] 目标视频: {target_video}")
    print(f"[Step 1] 目标输出: {save_path}")

    # B. 读取视频帧
    print(f"[Step 2] 正在读取视频帧 (Decord)...")
    try:
        vr = VideoReader(target_video, ctx=cpu(0))
        total_frames = len(vr)
        print(f"   - 视频总帧数: {total_frames}")
        
        # 采样 32 帧
        indices = np.linspace(0, total_frames - 1, 32).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        print(f"   - 采样数据形状 (Numpy): {frames.shape} (T, H, W, C)")
        
        pil_images = [Image.fromarray(frm) for frm in frames]
        print(f"   - 成功转换为 {len(pil_images)} 张 PIL 图片")
    except Exception as e:
        print(f"   [Error] 读取视频失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # C. 加载模型
    print(f"[Step 3] 正在加载模型 (EVA-ViT-G)...")
    try:
        model = create_eva_vit_g(img_size=224, drop_path_rate=0, use_checkpoint=False, precision="fp16")
        checkpoint = torch.load(EVA_CKPT_PATH, map_location="cpu")
        
        # 处理 key
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"   - 权重加载信息: {msg}")
        model = model.to(DEVICE)
        model.eval()
        print(f"   - 模型已移动到: {DEVICE}")
    except Exception as e:
        print(f"   [Error] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # D. 预处理
    print(f"[Step 4] 正在进行图像预处理...")
    try:
        processor = ManualProcessor(image_size=224)
        tensor_list = []
        for i, img in enumerate(pil_images):
            tensor_list.append(processor(img))
        
        input_tensor = torch.stack(tensor_list).to(DEVICE).half()
        print(f"   - 输入 Tensor 形状: {input_tensor.shape} (应为 [32, 3, 224, 224])")
        print(f"   - 数据类型: {input_tensor.dtype}")
    except Exception as e:
        print(f"   [Error] 预处理失败: {e}")
        traceback.print_exc()
        return

    # E. 模型推理
    print(f"[Step 5] 正在执行前向传播 (Forward Features)...")
    try:
        with torch.no_grad():
            features = model.forward_features(input_tensor)
        print(f"   [Success] 特征提取成功！")
        print(f"   - 输出特征形状: {features.shape} (应为 [32, 257, 1408])")
    except Exception as e:
        print(f"   [Error] 推理失败: {e}")
        print("   提示: 可能是显存不足 (OOM) 或维度不匹配")
        traceback.print_exc()
        return

    # F. 保存
    print(f"[Step 6] 正在保存文件...")
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(features.cpu(), save_path)
        print(f"   [Success] 文件已保存: {save_path}")
        print(f"   - 文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"   [Error] 保存失败: {e}")
        return

    print(f"\n=== 测试全部通过！请放心使用批量脚本。 ===")

if __name__ == "__main__":
    test_one_video()
