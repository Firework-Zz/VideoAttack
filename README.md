# VideoAttack

Official code release for:

> Hierarchical Safety Failures in Video Reasoning: A Task-Driven Representation Method

This repository contains the task-driven pixel-space attack, the task-agnostic
feature-disruption baseline, the representation-space diagnostic attack, and
the keyframe-aware perturbation strategy used in the paper. It also includes
the Qwen3-VL evaluation and matched-iteration control code added in the second
revision.

## Repository layout

- `main_pixel_malmm.py`: original MA-LMM pixel-space attack.
- `main_pixel_malmm_keyframe.py`: keyframe-aware pixel attack for MSRVTT-QA.
- `main_pixel_malmm_keyframe_msvd.py`: keyframe-aware pixel attack for MSVD-QA.
- `main_feature_*.py`: representation-space attack utilities.
- `main_feature_justask*.py`: JustAsk evaluation and feature-space attacks.
- `modern_qwen/`: Qwen3-VL attack, evaluation, and matched-control code.
- `MA-LMM/`, `FrozenBilm/`, `just-ask/`, `lavis/`: model and dataset support
  code used by the experiments.

Datasets, model weights, generated features, logs, and evaluation outputs are
not included.

## Installation

The base model experiments use the dependencies listed in `requirements.txt`.
Qwen3-VL has newer dependencies and can be installed separately:

```bash
pip install -r modern_qwen/requirements.txt
```

## Data and checkpoints

By default, paths are resolved relative to the repository root. The expected
layout is:

```text
data/
  msrvtt/videos/
  msvd/videos/
  MSVD-QA/
  nextqa/
  videomme/
checkpoints/
  MA-LMM/
    llm/vicuna-7b/
    pretrained/eva_vit_g.pth
    pretrained/instruct_blip_vicuna7b_trimmed.pth
```

All dataset and checkpoint arguments can be overridden from the command line.

## MA-LMM keyframe-aware attack

```bash
python main_pixel_malmm_keyframe.py \
  --base_dir data/msrvtt/ \
  --qa_csv just-ask/MSRVTT-QA/test.csv \
  --vicuna_path checkpoints/MA-LMM/llm/vicuna-7b/ \
  --eva_ckpt checkpoints/MA-LMM/pretrained/eva_vit_g.pth \
  --malmm_ckpt checkpoints/MA-LMM/pretrained/instruct_blip_vicuna7b_trimmed.pth \
  --num_keyframes 8 --steps 10
```

For MSVD-QA, use `main_pixel_malmm_keyframe_msvd.py` with the corresponding
dataset and QA paths.

## Qwen3-VL evaluation and attacks

```bash
python modern_qwen/qwen3vl_eval_attack.py \
  --dataset nextqa \
  --attack both \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --limit 100 \
  --max-frames 8 \
  --steps 10
```

The matched-iteration control between the task-driven pixel objective and the
task-agnostic feature-disruption objective is available in:

```bash
python modern_qwen/qwen3vl_taskagnostic_control.py \
  --dataset nextqa \
  --limit 100 \
  --steps 10 \
  --model-path Qwen/Qwen3-VL-8B-Instruct
```

If FlashAttention is unavailable, set `QWEN3VL_ATTN=eager`.

See `modern_qwen/README.md` for dataset arguments and sharded execution.

## Citation

Please cite the paper when using this code. The bibliography entry will be
updated after publication.
