#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL white-box video-QA evaluation + task-driven attack.

Datasets : msrvtt / msvd (open-ended), nextqa / videomme (multi-choice).

Attacks:
  * pixel          : PGD on the normalized ``pixel_values_videos`` tensor.
                     eps/alpha are expressed in the same normalized units as
                     the processor output, which keeps this first version
                     consistent with the smoke-test implementation.
  * representation : PGD on the merged visual-token embeddings that the
                     language model consumes. The perturbation is bounded by an
                     L_inf ball in that feature space.

Important: every item is decoded once (``read_video_frames``) and the same
frame tensor is used for the clean run and all attack variants, so the
comparison is fair and the number of video tokens never changes.

Sharded multi-GPU usage:
    WORLD_SIZE=4 RANK=0 CUDA_VISIBLE_DEVICES=0 python qwen3vl_eval_attack.py ...

Each rank writes its own JSONL; merge with merge_shards.py afterwards.
"""

import argparse
import copy
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_ranks(n: int) -> Tuple[List[int], int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    return [i for i in range(n) if i % world == rank], world, rank


def read_video_frames_pyav(path: str, max_frames: int) -> np.ndarray:
    """Fast uniform sampling with PyAV: seek to target timestamps and decode
    only the frames we keep (no full-video decode)."""
    import av

    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    wanted = max_frames or 8

    duration = stream.duration
    if not duration or duration <= 0:
        raise ValueError("stream duration unknown")
    time_base = float(stream.time_base)
    picks = np.linspace(0.0, 1.0, wanted)  # fractions of the clip
    frames = []
    for frac in picks:
        target_ts = int(frac * duration)
        container.seek(target_ts, stream=stream, backward=True, any_frame=False)
        got = None
        for frame in container.decode(stream):
            got = frame
            break
        if got is None:
            continue
        frames.append(np.asarray(got.to_image()))
    container.close()
    if not frames:
        raise ValueError(f"no decodable frames: {path}")
    return np.stack(frames)


def read_video_frames(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    want = max_frames or 8
    # Preferred: fast seek-style sampling with PyAV (decodes only what we keep).
    try:
        return read_video_frames_pyav(path, want)
    except Exception as e_pyav:
        pass
    # Fallback: torchvision full decode, then uniform subsample.
    import torchvision.io

    try:
        frames = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    except Exception:
        raise ValueError(f"unable to decode video with pyav/torchvision: {path}") from None
    if frames.numel() == 0:
        raise ValueError(f"empty video: {path}")
    if frames.shape[0] > want:
        idx = torch.linspace(0, frames.shape[0] - 1, want).long()
        frames = frames[idx]
    return frames.numpy()


def make_messages(question: str, answer: Optional[str] = None) -> List[dict]:
    user_content = [
        {"type": "video", "video": "<VIDEO>"},
        {"type": "text", "text": f"Question: {question}\nAnswer:"},
    ]
    messages = [{"role": "user", "content": user_content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def build_question_text(item: Dict) -> str:
    """Question prompt with the A/B/C/D option list for multiple-choice items.

    Open-ended items (MSVD/MSRVTT) keep the raw question; choice datasets get
    a deterministic option block so the model can answer with a letter and the
    clean/attack encodings always share the same text.
    """
    q = (item.get("question") or "").strip()
    choices = item.get("choices") or []
    if item.get("answer_letter") and choices:
        lines = []
        for i, c in enumerate(choices):
            letter = chr(ord("A") + i)
            text = re.sub(r"^[A-E][.)]\s*", "", str(c).strip(), flags=re.I)
            lines.append(f"{letter}. {text}")
        q = q.rstrip() + "\nOptions:\n" + "\n".join(lines)
    return q


def encode_conversation(processor, frames_tchw, question, answer, device,
                        add_generation_prompt):
    """Encode one video (T,C,H,W uint8 tensor) + text without qwen-vl-utils.
    The processor gets the same pre-sampled frames every time, so repeated
    calls are cheap and pixel values are identical across clean/attack runs."""
    messages = make_messages(question, answer)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    enc = processor(
        text=[text],
        videos=[frames_tchw],
        return_tensors="pt",
        padding=True,
        **({"cap_pixels_per_frame": True} if os.environ.get("QWEN3VL_CAP_PIXELS") == "1" else {}),
    )
    return {k: v.to(device) for k, v in enc.items()}


def prompt_enc(processor, frames_tchw, question, device) -> dict:
    return encode_conversation(processor, frames_tchw, question, None, device, True)


def full_enc(processor, frames_tchw, question, answer, device) -> dict:
    return encode_conversation(processor, frames_tchw, question, answer, device, False)


def numpy_to_tchw(video_np):
    """(T,H,W,3) uint8 -> (T,3,H,W) uint8 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(video_np)).permute(0, 3, 1, 2).contiguous()


def answer_labels(processor, full, prompt_len) -> torch.Tensor:
    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return labels


def visual_embeds(model, enc) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    # Encodings from the processor may keep integer-valued pixels; always feed
    # float values (the library casts to the vision dtype internally).
    pv_raw = enc["pixel_values_videos"]
    pv = pv_raw.float()
    if os.environ.get("QWEN3VL_DEBUG_VISUAL"):
        print("DBG pv_raw dtype:", pv_raw.dtype, "pv dtype:", pv.dtype, "model dtype:", model.dtype,
              "grid:", enc.get("video_grid_thw"), flush=True)
    vout = model.get_video_features(pv, enc.get("video_grid_thw"))
    if isinstance(vout, tuple):
        ve, deep = vout
    else:
        ve = vout.pooler_output
        deep = vout.hidden_states
    # Do NOT cast to input_ids.dtype (int64!). Cast to the LM embedding dtype.
    emb_dtype = model.model.get_input_embeddings().weight.dtype
    ve = torch.cat(ve, dim=0).to(device=enc["input_ids"].device, dtype=emb_dtype)
    if os.environ.get("QWEN3VL_DEBUG_VISUAL"):
        print("DBG ve dtype:", ve.dtype, "ve shape:", tuple(ve.shape),
              "deep dtypes:", [d.dtype for d in deep] if isinstance(deep, list) else None, flush=True)
    return ve, deep


def representation_attack(model, processor, frames_tchw, question, answer, device,
                          steps=10, eps=0.1, alpha=0.02, layer=0):
    """PGD on the *visual feature sequence* inside the vision tower (layer
    ``layer`` of ``model.visual.blocks``). The perturbation is added right
    before that transformer block and flows through the rest of the vision
    tower, the DeepStack branches and the LM, so the whole model forward is
    native (no manual embedding assembly).

    eps/alpha are in the L_inf sense over these hidden features.
    """
    full = full_enc(processor, frames_tchw, question, answer, device)
    prompt_len = prompt_enc(processor, frames_tchw, question, device)["input_ids"].shape[1]
    labels = answer_labels(processor, full, prompt_len)
    full["_labels"] = labels

    block = model.visual.blocks[layer]
    state = {"delta": None}

    def pre_hook(module, args):
        x = args[0]
        return (x + state["delta"].to(x.dtype),) + tuple(args[1:])

    # Prime the delta shape from the actual block input via a no-grad forward.
    captured = {}

    def capture_hook(module, args):
        captured["shape"] = tuple(args[0].shape)

    cap_handle = block.register_forward_pre_hook(capture_hook)
    with torch.no_grad():
        _ = model(**full, return_dict=True)
    cap_handle.remove()
    # Only the perturbation should require grad; all model inputs are frozen.
    for k, v in full.items():
        if isinstance(v, torch.Tensor):
            full[k] = v.detach()
    state["delta"] = torch.zeros(captured["shape"], dtype=torch.float32,
                                 device=full["input_ids"].device).requires_grad_(True)
    handle = block.register_forward_pre_hook(pre_hook)
    t0 = time.perf_counter()
    try:
        for _ in range(steps):
            model.zero_grad()
            if state["delta"].grad is not None:
                state["delta"].grad.zero_()
            loss = model(**full, labels=labels, return_dict=True).loss
            loss.backward()
            with torch.no_grad():
                if state["delta"].grad is None:
                    break
                state["delta"].data = state["delta"].data + alpha * state["delta"].grad.sign()
                state["delta"].data = torch.clamp(state["delta"].data, -eps, eps)
        elapsed = time.perf_counter() - t0
        delta = state["delta"].detach().clone()
    finally:
        handle.remove()
    return delta, elapsed


# --------------------------------------------------------------------------- #
# attacks
# --------------------------------------------------------------------------- #
def pixel_attack(model, processor, frames_tchw, question, answer, device,
                 steps=10, eps=8.0 / 255.0, alpha=2.0 / 255.0):
    full = full_enc(processor, frames_tchw, question, answer, device)
    prompt_len = prompt_enc(processor, frames_tchw, question, device)["input_ids"].shape[1]
    labels = answer_labels(processor, full, prompt_len)
    pv0 = full["pixel_values_videos"].float().detach()
    pv = pv0.clone().requires_grad_(True)
    full["pixel_values_videos"] = pv
    t0 = time.perf_counter()
    for _ in range(steps):
        model.zero_grad()
        if pv.grad is not None:
            pv.grad.zero_()
        loss = model(**full, labels=labels, return_dict=True).loss
        loss.backward()
        with torch.no_grad():
            if pv.grad is None:
                break
            pv.data = pv.data + alpha * pv.grad.sign()
            pv.data = torch.clamp(pv.data, pv0 - eps, pv0 + eps)
    elapsed = time.perf_counter() - t0
    return pv.detach(), elapsed


# --------------------------------------------------------------------------- #
# generation / evaluation
# --------------------------------------------------------------------------- #
def generate_answer(model, processor, enc) -> str:
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=48, do_sample=False)
    input_len = enc["input_ids"].shape[1]
    text = processor.batch_decode(
        gen[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text


def generate_answer_pixel_adv(model, processor, prompt_enc_clean, pv) -> str:
    adv = copy.deepcopy(prompt_enc_clean)
    n_clean = adv["pixel_values_videos"].shape[0]
    adv["pixel_values_videos"] = pv[:n_clean]
    return generate_answer(model, processor, adv)


def generate_with_feature_delta(model, processor, prompt_enc_clean, delta, layer=0):
    """Generate while adding a feature-space delta in the vision tower."""
    block = model.visual.blocks[layer]

    def pre_hook(module, args):
        x = args[0]
        return (x + delta.to(x.dtype),) + tuple(args[1:])

    handle = block.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            gen = model.generate(**prompt_enc_clean, max_new_tokens=48, do_sample=False)
    finally:
        handle.remove()
    input_len = prompt_enc_clean["input_ids"].shape[1]
    return processor.batch_decode(
        gen[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def load_model(model_path, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    attn = os.environ.get("QWEN3VL_ATTN", "flash_attention_2")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def is_correct(gen_text: str, item: Dict) -> Tuple[bool, str]:
    choices = item.get("choices") or []
    gold_letter = item.get("answer_letter") or ""
    gold = re.sub(r"\s+", " ", (item.get("answer_text") or "").strip().lower())
    norm = re.sub(r"\s+", " ", gen_text).strip().lower()
    if choices:
        m = re.search(r"\b([A-E])\b", gen_text.upper())
        if m and gold_letter:
            return m.group(1) == gold_letter, gen_text
        return (gold in norm) or (norm in gold), gen_text
    return (gold in norm) or (norm in gold), gen_text


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["msrvtt", "msvd", "nextqa", "videomme"], required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--attack", choices=["none", "pixel", "representation", "both"], default="none")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--pixel-eps", type=float, default=8.0 / 255.0)
    ap.add_argument("--pixel-alpha", type=float, default=2.0 / 255.0)
    ap.add_argument("--repr-eps", type=float, default=0.1)
    ap.add_argument("--repr-alpha", type=float, default=0.02)
    ap.add_argument("--out-dir", default="./outputs/qwen3vl_attack")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from qwen3vl_dataset_adapter import LOADERS

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = LOADERS[args.dataset](limit=args.limit, offset=args.offset)
    idxs, world, rank = parse_ranks(len(items))
    items = [items[i] for i in idxs]
    print(f"[rank {rank}/{world}] {args.dataset}: {len(items)} items, device={device}", flush=True)
    if not items:
        return

    model, processor = load_model(args.model_path, device)
    out_path = os.path.join(args.out_dir, f"{args.tag}_r{rank}_of_{world}.jsonl")
    results = []

    for it in items:
        rec = dict(it)
        try:
            frames_np = read_video_frames(it["video_path"], max_frames=args.max_frames)
            frames_tchw = numpy_to_tchw(frames_np)
            q_prompt = build_question_text(it)
            gold_answer = (it.get("answer_letter") or "").strip() or (it.get("answer_text") or "")
            clean_enc = prompt_enc(processor, frames_tchw, q_prompt, device)
            clean_text = generate_answer(model, processor, clean_enc)
            ok_clean, _ = is_correct(clean_text, it)
            rec["clean_pred"] = clean_text
            rec["clean_correct"] = ok_clean

            if args.attack in ("pixel", "both"):
                pv, dt = pixel_attack(
                    model, processor, frames_tchw, q_prompt, gold_answer,
                    device, steps=args.steps, eps=args.pixel_eps, alpha=args.pixel_alpha,
                )
                adv_text = generate_answer_pixel_adv(model, processor, clean_enc, pv)
                ok_adv, _ = is_correct(adv_text, it)
                rec["pixel_pred"] = adv_text
                rec["pixel_correct"] = ok_adv
                rec["pixel_time_s"] = dt

            if args.attack in ("representation", "both"):
                delta, dt = representation_attack(
                    model, processor, frames_tchw, q_prompt, gold_answer,
                    device, steps=args.steps, eps=args.repr_eps, alpha=args.repr_alpha,
                )
                adv_text = generate_with_feature_delta(model, processor, clean_enc, delta)
                ok_adv, _ = is_correct(adv_text, it)
                rec["repr_pred"] = adv_text
                rec["repr_correct"] = ok_adv
                rec["repr_time_s"] = dt

            results.append(rec)
            flags = "T" if ok_clean else "F"
            if "pixel_correct" in rec:
                flags += "/" + ("T" if rec["pixel_correct"] else "F")
            if "repr_correct" in rec:
                flags += "/" + ("T" if rec["repr_correct"] else "F")
            print(f"[{it['dataset']}] {it['video_id']} acc={flags} | {clean_text[:50]}", flush=True)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            results.append(rec)
            print(f"[error] {it['video_id']}: {type(e).__name__}: {e}", flush=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    summary = {
        "dataset": args.dataset,
        "attack": args.attack,
        "tag": args.tag,
        "rank": rank,
        "world": world,
        "n": n,
        "clean_acc": sum(1 for r in results if r.get("clean_correct")) / n if n else 0.0,
        "pixel_acc": sum(1 for r in results if r.get("pixel_correct")) / n if n else 0.0,
        "repr_acc": sum(1 for r in results if r.get("repr_correct")) / n if n else 0.0,
    }
    with open(os.path.join(args.out_dir, f"summary_{args.tag}_r{rank}_of_{world}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
