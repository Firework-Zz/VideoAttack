import cv2
import numpy as np
from typing import List

def sample_frames(video_path: str, fps: int = 10) -> List[np.ndarray]:
    frames = []
    
    # 1. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 2. 检查视频是否成功打开
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {video_path}")
        return frames

    # 3. 获取视频的原始帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print(f"[WARN] 无法读取视频 {video_path} 的原始帧率, 将默认使用 30 FPS 进行计算。")
        original_fps = 30  # 提供一个合理的默认值

    # 4. 计算采样间隔 (stride)
    # 例如：原始是30fps，目标是10fps，则 stride = round(30/10) = 3，即每3帧取1帧。
    # max(1, ...) 确保我们至少每帧都看，避免 stride 为0。
    stride = max(1, round(original_fps / fps))
    
    frame_idx = 0
    while cap.isOpened():
        # 5. 读取一帧
        ret, frame = cap.read()
        
        # 如果 ret 是 False，说明视频已经读完
        if not ret:
            break
        
        # 6. 判断是否是我们要采样的帧
        if frame_idx % stride == 0:
            # OpenCV 读取的帧是 BGR 格式，需要转换为 RGB 格式
            # 这是因为大多数深度学习模型（如PyTorch的预训练模型）都使用RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        
        frame_idx += 1

    # 7. 释放视频捕获对象，这是一个好习惯
    cap.release()
    
    return frames

