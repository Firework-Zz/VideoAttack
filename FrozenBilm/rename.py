#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

TARGET_DIR = "/data0/data/msrvtt/malmm_to_frozenbilm"

def extract_id(name: str):
    m = re.search(r'(\d+)', name)
    return m.group(1) if m else None

def main():
    files = sorted(os.listdir(TARGET_DIR))
    renamed = 0
    skipped = 0

    for fn in files:
        src = os.path.join(TARGET_DIR, fn)
        if not os.path.isfile(src):
            continue

        # 已经是目标格式的，跳过
        if fn.endswith("_adv.pt"):
            skipped += 1
            continue

        base, ext = os.path.splitext(fn)
        if ext.lower() not in [".pt", ".pth"]:
            continue

        vid = extract_id(fn)
        if vid is None:
            skipped += 1
            continue

        dst_name = f"video{vid}_adv.pt"
        dst = os.path.join(TARGET_DIR, dst_name)

        if os.path.exists(dst):
            print(f"[SKIP] target exists: {dst_name}")
            skipped += 1
            continue

        print(f"[RENAME] {fn} -> {dst_name}")
        os.rename(src, dst)
        renamed += 1

    print(f"\n[DONE] renamed={renamed}, skipped={skipped}, dir={TARGET_DIR}")

if __name__ == "__main__":
    main()
