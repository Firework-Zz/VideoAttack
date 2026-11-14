#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset adapters for Qwen3-VL video-QA attacks/evaluation.

Supported sources (all local on the L20 server):
  * msrvtt  : ./data/msrvtt/videos/*.mp4 + JustAsk MSRVTT-QA/test.csv
  * msvd    : ./data/msvd/videos/*.avi  + ./data/msvd/annotation/*.csv
  * nextqa  : ./data/nextqa/test.csv    + ./data/nextqa/videos/<4-prefix>/<id>.mp4
  * videomme: ./data/videomme/test-00000-of-00001.parquet
              + ./data/videomme/videos/<youtube-id>.mp4

Each item exposes:
    video_id, video_path, question, answer_letter, choices, answer_text, task_type

The output format is JSONL so that multi-GPU shards can be merged later.
"""

import argparse
import ast
import csv
import json
import os
import re
import sys


def find_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _parse_videomme_options(raw):
    """Parse VideoMME's options field into a list of plain option texts.

    The parquet stores options as a Python-list literal (e.g. "['A. Apples.',
    'B. Candles.', ...]"); a few rows may use line-separated text instead.
    Letter prefixes are stripped so the eval script can renumber A/B/C/D
    deterministically.
    """
    raw = (raw or "").strip()
    entries = []
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            entries = parsed
        else:
            entries = []
    if not entries and raw:
        entries = [s for s in re.split(r"(?=[A-E]\.\s)", raw) if s.strip()]
    out = []
    seen = set()
    for ent in entries:
        s = str(ent).strip()
        if not s:
            continue
        m = re.match(r"^([A-E])[.)]\s*(.*)$", s, flags=re.S)
        if m:
            letter = m.group(1)
            text = m.group(2).strip()
        else:
            letter = chr(ord("A") + len(out))
            text = s
        if letter in seen:
            continue
        seen.add(letter)
        out.append(text)
    return out


def load_msrvtt(video_dir=None, csv_path=None, limit=None, offset=0, name="msrvtt"):
    """JustAsk MSRVTT-QA test.csv -> open-ended QA (first answer text)."""
    video_dir = video_dir or "./data/msrvtt/videos"
    csv_path = csv_path or "./just-ask/MSRVTT-QA/test.csv"
    items = []
    seen = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not (vid and q and a):
                continue
            if vid in seen:
                continue
            seen.add(vid)
            for cand in (os.path.join(video_dir, f"{vid}.mp4"),
                         os.path.join(video_dir, f"video{vid}.mp4")):
                if os.path.exists(cand):
                    items.append({
                        "dataset": name,
                        "video_id": vid,
                        "video_path": cand,
                        "question": q,
                        "answer_letter": "",
                        "choices": [],
                        "answer_text": a,
                        "task_type": "open",
                    })
                    break
    return items[offset:(None if limit is None else offset + limit)]


def load_msvd(video_dir=None, qa_root=None, mapping=None, limit=None, offset=0, name="msvd"):
    """JustAsk MSVD-QA test.csv -> open-ended QA (first answer text)."""
    video_dir = video_dir or "./data/msvd/videos"
    qa_root = qa_root or "./data/MSVD-QA"
    mapping = mapping or os.path.join(qa_root, "youtube_mapping.txt")
    csv_path = os.path.join(qa_root, "test.csv")

    # clip name -> short numeric id ("jfrrO5K_vKM_55_65" -> "1451")
    id2clip = {}
    if os.path.exists(mapping):
        with open(mapping, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    digits = "".join(ch for ch in parts[1] if ch.isdigit())
                    if digits:
                        id2clip[digits] = parts[0]
    items = []
    seen = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not (vid and q and a):
                continue
            if vid in seen:
                continue
            seen.add(vid)
            clip = id2clip.get(vid)
            if not clip:
                continue
            vpath = find_existing([
                os.path.join(video_dir, f"{clip}.avi"),
                os.path.join(video_dir, f"{clip}.mp4"),
                os.path.join(video_dir, f"{clip}.webm"),
            ])
            if not vpath:
                continue
            items.append({
                "dataset": name,
                "video_id": vid,
                "video_path": vpath,
                "question": q,
                "answer_letter": "",
                "choices": [],
                "answer_text": a,
                "task_type": "open",
            })
    return items[offset:(None if limit is None else offset + limit)]


def load_nextqa(csv_path=None, video_dir=None, map_path=None,
                limit=None, offset=0, name="nextqa"):
    """NExT-QA test.csv -> MCQA with 5 choices.

    ``map_vid_vidorID.json`` provides the relative video path for every clip
    (e.g. '2574374895' -> '1101/2574374895'), so no prefix guessing is needed.
    """
    csv_path = csv_path or "./data/nextqa/test.csv"
    video_dir = video_dir or "./data/nextqa/videos"
    map_path = map_path or "./data/nextqa/map_vid_vidorID.json"
    with open(map_path, "r", encoding="utf-8") as f:
        vid2rel = json.load(f)
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video") or "").strip()
            q = (row.get("question") or "").strip()
            a_idx = (row.get("answer") or "").strip()
            choices = [(row.get(f"a{i}") or "").strip() for i in range(5)]
            choices = [c for c in choices if c]
            if not (vid and q and a_idx and choices):
                continue
            try:
                ai = int(a_idx)
            except ValueError:
                ai = -1
            if not (0 <= ai < len(choices)):
                continue
            rel = vid2rel.get(vid)
            cands = []
            if rel:
                cands.append(os.path.join(video_dir, rel + ".mp4"))
            cands.append(os.path.join(video_dir, f"{vid}.mp4"))
            vpath = find_existing(cands)
            if not vpath:
                continue
            items.append({
                "dataset": name,
                "video_id": vid,
                "video_path": vpath,
                "question": q,
                "answer_letter": "ABCDE"[ai],
                "choices": choices,
                "answer_text": choices[ai],
                "task_type": row.get("type", "MC"),
            })
    return items[offset:(None if limit is None else offset + limit)]


def load_videomme(parquet_path=None, video_dir=None, limit=None, offset=0, name="videomme"):
    """VideoMME test parquet -> MCQA (official 2700-row subset)."""
    parquet_path = parquet_path or "./data/videomme/test-00000-of-00001.parquet"
    video_dir = video_dir or "./data/videomme/videos"
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas is required to read the VideoMME parquet") from e
    df = pd.read_parquet(parquet_path)
    items = []
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        yid = str(row["videoID"])
        q = str(row["question"]).strip()
        ans = str(row["answer"]).strip().upper()
        options = str(row["options"]).strip()
        vpath = os.path.join(video_dir, f"{yid}.mp4")
        if not os.path.exists(vpath) or not q or not options:
            continue
        choices = _parse_videomme_options(options)
        if not choices:
            continue
        answer_letter = ans[0] if ans and ans[0] in "ABCDE" else ""
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        try:
            ai = letters.index(answer_letter) if answer_letter else -1
        except ValueError:
            ai = -1
        if ai < 0:
            # Row is unusable without a valid gold letter for choice matching.
            continue
        items.append({
            "dataset": name,
            "video_id": vid,
            "video_path": vpath,
            "question": q,
            "answer_letter": answer_letter,
            "choices": choices,
            "answer_text": choices[ai],
            "task_type": str(row.get("task_type", "MC")),
        })
    return items[offset:(None if limit is None else offset + limit)]


LOADERS = {
    "msrvtt": load_msrvtt,
    "msvd": load_msvd,
    "nextqa": load_nextqa,
    "videomme": load_videomme,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = LOADERS[args.dataset](limit=args.limit, offset=args.offset)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{args.dataset}: {len(items)} items -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
