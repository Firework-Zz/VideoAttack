#!/bin/bash

# ==================== 1. 路径配置 ====================
# MA-LMM 权重 (本地路径)
checkpoint_path="/data0/pretrained/MA-LMM/saved_model/MSRVTT_qa/checkpoint_best.pth"

# Vicuna LLM 本地路径 (用于加载 Tokenizer)
vicuna_path="/data0/pretrained/MA-LMM/llm/vicuna-7b/"

# ==================== 2. 环境设置 ====================
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# 强制离线模式
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ==================== 3. 运行命令 ====================
echo "开始运行离线测试..."
echo "使用本地权重: ${checkpoint_path}"

torchrun --nproc_per_node=4 \
    --master_port=29505 \
    train.py \
    --cfg-path lavis/projects/malmm/qa_msrvtt.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.num_query_token 32 \
    model.num_frames 20 \
    model.llm_model ${vicuna_path} \
    model.pretrained ${checkpoint_path} \
    run.batch_size_eval 32 \
    run.num_workers 4 \
    run.seed 42 \
    run.evaluate True \
    run.report_metric True \
    run.valid_splits "['test']" \
    run.prefix test
    # [删除] run.resume_ckpt_path 这一行已经被删除了
