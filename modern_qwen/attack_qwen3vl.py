#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White-box task-driven attack for Qwen3-VL on video QA.

Two attack levels are implemented:
  * pixel-space PGD on the video visual inputs (pixel_values_videos)
  * representation-space PGD on the visual-token features emitted by the
    vision tower (via a forward hook)

This is a first runnable version. The pixel-space attack currently operates on
the already-normalized ``pixel_values_videos`` tensor. For a strict 8/255
raw-pixel threat model, the preprocessing path should be made differentiable
before final experiments.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def read_video_frames(path: str, max_frames: int = 8) -> np.ndarray:
    import torchvision.io

    frames, _, info = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")
    if frames.numel() == 0:
        raise ValueError(f"Empty video: {path}")
    if frames.shape[0] > max_frames:
        idx = torch.linspace(0, frames.shape[0] - 1, max_frames).long()
        frames = frames[idx]
    return frames.numpy()


def build_inputs(
    processor,
    video_np: np.ndarray,
    question: str,
    answer: Optional[str],
    device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
    user_content = [
        {"type": "video", "video": video_np},
        {"type": "text", "text": f"Question: {question}\nAnswer:"},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_images, prompt_videos = process_vision_info(prompt_messages)
    prompt_enc = processor(
        text=[prompt_text],
        images=prompt_images,
        videos=prompt_videos,
        return_tensors="pt",
        padding=True,
    )
    prompt_len = int(prompt_enc["input_ids"].shape[1])

    full_messages = prompt_messages + [{"role": "assistant", "content": answer or ""}]
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_images, full_videos = process_vision_info(full_messages)
    inputs = processor(
        text=[full_text],
        images=full_images,
        videos=full_videos,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return inputs, labels, prompt_len


def forward_loss(model, inputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    outputs = model(**inputs, labels=labels, return_dict=True)
    return outputs.loss


def project_to_eps(x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(x, x0 - eps, x0 + eps)


def pixel_attack(
    model,
    processor,
    video_np: np.ndarray,
    question: str,
    answer: str,
    device,
    steps: int = 10,
    eps: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
) -> Tuple[Dict[str, torch.Tensor], float]:
    inputs, labels, _ = build_inputs(processor, video_np, question, answer, device)
    if "pixel_values_videos" not in inputs:
        raise RuntimeError("No pixel_values_videos in encoded inputs")
    pv = inputs["pixel_values_videos"].float().detach()
    pv0 = pv.clone()
    pv = pv.clone().requires_grad_(True)
    inputs["pixel_values_videos"] = pv

    t0 = time.perf_counter()
    for _ in range(steps):
        model.zero_grad()
        if pv.grad is not None:
            pv.grad.zero_()
        loss = forward_loss(model, inputs, labels)
        loss.backward()
        with torch.no_grad():
            grad = pv.grad
            if grad is None:
                break
            pv.data = pv.data + alpha * grad.sign()
            pv.data = project_to_eps(pv.data, pv0, eps)
    elapsed = time.perf_counter() - t0
    inputs["pixel_values_videos"] = pv.detach()
    return inputs, elapsed


def representation_attack(
    model,
    processor,
    video_np: np.ndarray,
    question: str,
    answer: str,
    device,
    steps: int = 10,
    eps: float = 0.1,
    alpha: float = 0.02,
) -> Tuple[Dict[str, torch.Tensor], float]:
    inputs, labels, _ = build_inputs(processor, video_np, question, answer, device)

    visual_module = None
    for name, mod in model.named_modules():
        if name.endswith(".visual") or name == "visual":
            visual_module = mod
            break
    if visual_module is None:
        raise RuntimeError("Could not locate the vision tower for representation attack")

    delta = None
    handle = None

    def hook(module, args, output):
        nonlocal delta
        return output + delta

    handle = visual_module.register_forward_hook(hook)
    t0 = time.perf_counter()
    for _ in range(steps):
        if delta is None:
            with torch.no_grad():
                _ = model(**inputs, labels=labels, return_dict=True)
            raise RuntimeError("Hook did not observe visual output")
        model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()
        loss = forward_loss(model, inputs, labels)
        loss.backward()
        with torch.no_grad():
            if delta.grad is None:
                break
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -eps, eps)
    elapsed = time.perf_counter() - t0
    if handle is not None:
        handle.remove()
    return inputs, elapsed


def generate_answer(model, processor, inputs: Dict[str, torch.Tensor]) -> str:
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    text = processor.batch_decode(
        gen[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--csv", default="./just-ask/MSRVTT-QA/test.csv")
    parser.add_argument("--video-dir", default="./data/msrvtt/videos")
    parser.add_argument("--attack", choices=["pixel", "representation", "both"], default="pixel")
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--pixel-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--pixel-alpha", type=float, default=2.0 / 255.0)
    parser.add_argument("--repr-eps", type=float, default=0.1)
    parser.add_argument("--repr-alpha", type=float, default=0.02)
    parser.add_argument("--out-dir", default="./outputs/qwen3vl_attack")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    import csv

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows = rows[: args.max_videos]

    for row in rows:
        vid = (row.get("video_id") or row.get("video") or "").strip()
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not (vid and q and a):
            continue
        vpath = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(vpath):
            # MSRVTT naming may include the 'video' prefix.
            alt = os.path.join(args.video_dir, f"video{vid}.mp4")
            vpath = alt if os.path.exists(alt) else vpath
        if not os.path.exists(vpath):
            continue
        try:
            video_np = read_video_frames(vpath, max_frames=args.max_frames)
            if args.attack in ("pixel", "both"):
                adv_inputs, elapsed = pixel_attack(
                    model,
                    processor,
                    video_np,
                    q,
                    a,
                    device,
                    steps=args.steps,
                    eps=args.pixel_eps,
                    alpha=args.pixel_alpha,
                )
                clean_inputs, _, _ = build_inputs(processor, video_np, q, a, device)
                print(vid, "pixel", generate_answer(model, processor, clean_inputs), "->", generate_answer(model, processor, adv_inputs), f"{elapsed:.2f}s", flush=True)
            if args.attack in ("representation", "both"):
                adv_inputs, elapsed = representation_attack(
                    model,
                    processor,
                    video_np,
                    q,
                    a,
                    device,
                    steps=args.steps,
                    eps=args.repr_eps,
                    alpha=args.repr_alpha,
                )
                print(vid, "repr", generate_answer(model, processor, adv_inputs), f"{elapsed:.2f}s", flush=True)
        except Exception as e:
            print(f"[error] {vid}: {e}", flush=True)


if __name__ == "__main__":
    main()
