#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import cv2
import clip
from tqdm import tqdm

# ================== 配置 ==================
FRAMES_ROOT = "/data0/data/msrvtt/frames_attack"   # frames_attack 根目录
OUTPUT_DIR  = "/data0/data/msrvtt/malmm_to_frozenbilm"

NUM_FRAMES = 10
FOLDER_PREFIX = "video"     # videoxxxx
SKIP_EXISTING = True

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# =========================================


def load_frames_from_folder(folder, num_frames=10, size=224):
    files = [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]
    files.sort()

    if len(files) == 0:
        # 全黑帧兜底
        frames = [np.zeros((size, size, 3), np.uint8) for _ in range(num_frames)]
    else:
        vlen = len(files)
        indices = np.linspace(0, vlen - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            img_path = os.path.join(folder, files[int(idx)])
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((size, size, 3), np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (size, size))
            frames.append(img)

    np_frames = np.stack(frames)                 # (T,H,W,C)
    tensor = torch.from_numpy(np_frames).float() # float32
    tensor = tensor.permute(0, 3, 1, 2)           # (T,C,H,W)
    return tensor.unsqueeze(0) / 255.0            # (1,T,C,H,W)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Using device: {DEVICE}")
    print("[INFO] Loading CLIP ViT-L/14 ...")

    # === CLIP：完全按照你 worker 的方式 ===
    clip_model, _ = clip.load("ViT-L/14", device=DEVICE, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=DEVICE
    ).view(1, 1, 3, 1, 1)

    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=DEVICE
    ).view(1, 1, 3, 1, 1)

    folders = sorted(os.listdir(FRAMES_ROOT))

    for name in tqdm(folders, desc="Extracting"):
        if FOLDER_PREFIX and not name.startswith(FOLDER_PREFIX):
            continue

        folder_path = os.path.join(FRAMES_ROOT, name)
        if not os.path.isdir(folder_path):
            continue

        save_path = os.path.join(OUTPUT_DIR, f"{name}.pth")
        if SKIP_EXISTING and os.path.exists(save_path):
            continue

        frames = load_frames_from_folder(folder_path, NUM_FRAMES).to(DEVICE)

        with torch.no_grad():
            frames_norm = (frames - mean) / std
            b, t, c, h, w = frames_norm.shape
            inputs = frames_norm.view(-1, c, h, w)

            feats = clip_model.encode_image(inputs).float()
            feats = feats.view(t, -1).cpu()   # (T, 768)

        torch.save(feats, save_path)

    print(f"[DONE] Features saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
