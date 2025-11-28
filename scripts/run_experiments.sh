#!/bin/bash
set -e  # 遇到错误立即退出

# ================= 配置区域 =================
# 请根据实际情况修改模型路径
MODEL_PATH="Qwen/QwQ-32B" 
# 目标数据集
DATASET="aime2024"
# 输出目录
OUTPUT_DIR="./outputs"
# GPU数量
NUM_GPUS=8

# 公共参数 (参考 README 的 Baseline 设置)
COMMON_ARGS="--dataset $DATASET \
--model_name $MODEL_PATH \
--num_gpus $NUM_GPUS \
--output_dir $OUTPUT_DIR \
--max_generated_tokens 32768 \
--temperature 0.6 \
--top_p 0.95 \
--top_k 30 \
--min_p 0.001 \
--after_thinking_temperature 0.6 \
--after_thinking_top_p 0.95 \
--after_thinking_top_k 30 \
--mem_fraction_static 0.8 \
--use_llm_judge \
--judge_model_name gpt-4.1-2025-04-14"
# ===========================================

echo "开始运行实验..."
mkdir -p $OUTPUT_DIR

# -------------------------------------------
# 实验 1: Discrete CoT (基线)
# -------------------------------------------
echo "[1/3] Running Discrete CoT..."
python run_sglang_softthinking.py $COMMON_ARGS \
    --max_topk 10 \
    --num_samples 16
    # 注意: 不加 --enable_soft_thinking 即为离散模式

# -------------------------------------------
# 实验 2: Original Soft Thinking (全程软思考 + 早停)
# -------------------------------------------
echo "[2/3] Running Original Soft Thinking..."
# 参考 README 的参数推荐
python run_sglang_softthinking.py $COMMON_ARGS \
    --enable_soft_thinking \
    --max_topk 10 \
    --early_stopping_entropy_threshold 0.01 \
    --early_stopping_length_threshold 256 \
    --num_samples 1

# -------------------------------------------
# 实验 3: Entropy Triggered Soft Thinking (动态触发)
# -------------------------------------------
echo "[3/3] Running Entropy Triggered Soft Thinking..."
# 示例参数：熵超过 2.5 时触发，思考 3 步，最多触发 10 次
# 注意：这里我们将原有的早停阈值设为 0，以完全依赖"固定步数"逻辑，或者你可以保留它作为双重保险

python run_sglang_softthinking.py $COMMON_ARGS \
    --enable_soft_thinking \
    --max_topk 10 \
    --soft_thinking_trigger_entropy 0.5 \
    --soft_thinking_steps 3 \
    --max_soft_thinking_triggers 10 \
    --early_stopping_entropy_threshold 0.0 \
    --num_samples 1
# soft_thinking_trigger_entropy：熵阈值，设为-1.0 则不触发
# soft_thinking_steps：固定思考步数
# max_soft_thinking_triggers：单次请求的最大触发次数

echo "所有实验完成！结果保存在 $OUTPUT_DIR/results/$DATASET 中。"