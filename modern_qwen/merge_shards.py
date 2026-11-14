#!/usr/bin/env python3
"""Merge per-rank JSONL shards written by qwen3vl_eval_attack.py into one file."""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pattern = os.path.join(args.dir, f"{args.tag}_r*_of_*.jsonl")
    files = sorted(glob.glob(pattern))
    rows, errors = [], 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "error" in r:
                    errors += 1
                rows.append(r)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    agg = {
        "files": files,
        "n": n,
        "errors": errors,
        "clean_acc": sum(1 for r in rows if r.get("clean_correct")) / n if n else 0.0,
        "pixel_acc": sum(1 for r in rows if r.get("pixel_correct")) / n if n else 0.0,
        "repr_acc": sum(1 for r in rows if r.get("repr_correct")) / n if n else 0.0,
    }
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
