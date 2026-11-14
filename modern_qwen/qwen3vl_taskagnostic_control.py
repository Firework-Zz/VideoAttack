#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL matched-iteration control: task-driven pixel vs task-agnostic pixel.

Task-agnostic objective: feature-disruption PGD that maximizes the cosine
distance between clean and adversarial visual-token embeddings, with no answer
supervision. Task-driven objective reuses qwen3vl_eval_attack.pixel_attack
(answer NLL, teacher forcing). Both run with the same number of PGD steps and
the same 8/255 pixel budget, so the results directly support the R2.4
matched-iteration comparison on a modern Video LLM.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nextqa", "videomme", "msrvtt", "msvd"], required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--items-json", type=str, default=None)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--agnostic-steps", type=int, default=None)
    ap.add_argument("--pixel-eps", type=float, default=8.0 / 255.0)
    ap.add_argument("--pixel-alpha", type=float, default=2.0 / 255.0)
    ap.add_argument("--out-dir", default="./outputs/qwen3vl_attack")
    ap.add_argument("--tag", default="taskagnostic_control")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    from qwen3vl_eval_attack import (
        build_question_text,
        generate_answer,
        generate_answer_pixel_adv,
        is_correct,
        load_model,
        numpy_to_tchw,
        parse_ranks,
        pixel_attack,
        prompt_enc,
        read_video_frames,
    )
    from qwen3vl_dataset_adapter import LOADERS

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agnostic_steps = args.agnostic_steps or args.steps

    if args.items_json:
        with open(args.items_json, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        items = items[args.offset:] if args.offset else items
        if args.limit:
            items = items[: args.limit]
    else:
        items = LOADERS[args.dataset](limit=args.limit, offset=args.offset)
    idxs, world, rank = parse_ranks(len(items))
    items = [items[i] for i in idxs]
    print(f"[rank {rank}/{world}] {args.dataset}: {len(items)} items, "
          f"steps={args.steps} agnostic_steps={agnostic_steps}", flush=True)
    if not items:
        return

    model, processor = load_model(args.model_path, device)
    out_path = os.path.join(args.out_dir, f"{args.tag}_r{rank}_of_{world}.jsonl")
    results = []

    def visual_token_feats(pv, grid):
        vout = model.get_video_features(pv.float(), grid)
        if isinstance(vout, tuple):
            ve = vout[0]
        else:
            ve = vout.pooler_output
        ve = torch.cat(ve, dim=0)
        emb_dtype = model.model.get_input_embeddings().weight.dtype
        return ve.to(device=device, dtype=emb_dtype)

    def agnostic_pixel_attack(frames_tchw, q_prompt):
        enc = prompt_enc(processor, frames_tchw, q_prompt, device)
        pv0 = enc["pixel_values_videos"].float().detach()
        grid = enc["video_grid_thw"]
        with torch.no_grad():
            clean_feats = visual_token_feats(pv0, grid).detach()
        clean_norm = F.normalize(clean_feats.reshape(-1, clean_feats.shape[-1]), p=2, dim=-1)

        pv = pv0.clone().requires_grad_(True)
        t0 = time.perf_counter()
        for _ in range(agnostic_steps):
            model.zero_grad()
            if pv.grad is not None:
                pv.grad.zero_()
            adv_feats = visual_token_feats(pv, grid)
            adv_norm = F.normalize(adv_feats.reshape(-1, adv_feats.shape[-1]), p=2, dim=-1)
            loss = -torch.einsum("ij,ij->i", clean_norm, adv_norm).mean()
            loss.backward()
            with torch.no_grad():
                if pv.grad is None:
                    break
                pv.data = pv.data + args.pixel_alpha * pv.grad.sign()
                pv.data = torch.clamp(pv.data, pv0 - args.pixel_eps, pv0 + args.pixel_eps)
        elapsed = time.perf_counter() - t0
        return pv.detach(), elapsed

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

            pv_td, td_time = pixel_attack(
                model, processor, frames_tchw, q_prompt, gold_answer,
                device, steps=args.steps, eps=args.pixel_eps, alpha=args.pixel_alpha,
            )
            td_text = generate_answer_pixel_adv(model, processor, clean_enc, pv_td)
            ok_td, _ = is_correct(td_text, it)
            rec["task_pixel_pred"] = td_text
            rec["task_pixel_correct"] = ok_td
            rec["task_pixel_time_s"] = td_time

            pv_ag, ag_time = agnostic_pixel_attack(frames_tchw, q_prompt)
            ag_text = generate_answer_pixel_adv(model, processor, clean_enc, pv_ag)
            ok_ag, _ = is_correct(ag_text, it)
            rec["agnostic_pixel_pred"] = ag_text
            rec["agnostic_pixel_correct"] = ok_ag
            rec["agnostic_pixel_time_s"] = ag_time

            results.append(rec)
            print(
                f"[{it['dataset']}] {it['video_id']} "
                f"clean={'T' if ok_clean else 'F'} task={'T' if ok_td else 'F'} "
                f"agnostic={'T' if ok_ag else 'F'}",
                flush=True,
            )
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            results.append(rec)
            print(f"[error] {it['video_id']}: {type(e).__name__}: {e}", flush=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if os.environ.get("QWEN3VL_EMPTY_CACHE") == "1":
            torch.cuda.empty_cache()

    n = len(results)
    summary = {
        "dataset": args.dataset,
        "tag": args.tag,
        "rank": rank,
        "world": world,
        "n": n,
        "clean_acc": sum(1 for r in results if r.get("clean_correct")) / n if n else 0.0,
        "task_pixel_acc": sum(1 for r in results if r.get("task_pixel_correct")) / n if n else 0.0,
        "agnostic_pixel_acc": sum(1 for r in results if r.get("agnostic_pixel_correct")) / n if n else 0.0,
        "task_pixel_drop": None,
        "agnostic_pixel_drop": None,
    }
    if n:
        summary["task_pixel_drop"] = summary["clean_acc"] - summary["task_pixel_acc"]
        summary["agnostic_pixel_drop"] = summary["clean_acc"] - summary["agnostic_pixel_acc"]
    with open(os.path.join(args.out_dir, f"summary_{args.tag}_r{rank}_of_{world}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
