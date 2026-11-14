# Qwen3-VL experiments

This directory contains the Qwen3-VL-8B-Instruct evaluation, attack, and
matched-iteration control code.

## Files

- `qwen3vl_eval_attack.py`: clean evaluation plus task-driven pixel-space and
  representation-space attacks.
- `qwen3vl_taskagnostic_control.py`: matched task-driven versus task-agnostic
  pixel-space comparison.
- `qwen3vl_dataset_adapter.py`: adapters for MSRVTT-QA, MSVD-QA, NExT-QA, and
  VideoMME.
- `attack_qwen3vl.py`: compact attack example.
- `merge_shards.py`: merge JSONL outputs from sharded runs.

## Datasets

The adapters expect the following relative layout:

```text
data/
  msrvtt/videos/
  msvd/videos/
  MSVD-QA/
  nextqa/test.csv
  nextqa/videos/
  nextqa/map_vid_vidorID.json
  videomme/test-00000-of-00001.parquet
  videomme/videos/
```

MSRVTT-QA uses `just-ask/MSRVTT-QA/test.csv` by default. All paths can be
overridden in Python by passing the corresponding arguments to the loader.

## Environment

```bash
pip install -r requirements.txt
```

Set `QWEN3VL_ATTN=eager` if FlashAttention is unavailable.

## Sharded evaluation

```bash
WORLD_SIZE=2 RANK=0 CUDA_VISIBLE_DEVICES=0 \
python qwen3vl_eval_attack.py --dataset videomme --attack both

WORLD_SIZE=2 RANK=1 CUDA_VISIBLE_DEVICES=1 \
python qwen3vl_eval_attack.py --dataset videomme --attack both

python merge_shards.py --dataset videomme --out-path outputs/qwen3vl
```

The scripts use a fixed eight-frame input and the same decoded frame tensor
for clean and adversarial evaluation unless command-line options specify
otherwise.
