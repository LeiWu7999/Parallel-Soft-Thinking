# 运行：  bash scripts/st/qwen2_5_7b_st_lora_math.sh
#!/usr/bin/env bash
set -euo pipefail

# 这个脚本用于快速验证：
# - base model + tokenizer 都使用 Qwen2.5-7B-Instruct 原版
# - 加载你训练得到的 token-wise gated LoRA checkpoint（lora_state.pt + lora_config.json）
# - 使用 soft-thinking 推理，并用 lora_mode=split 在 soft-thinking 阶段才启用 LoRA

# 可选：如果你要用 LLM Judge（run_sglang_softthinking.py 里有 --use_llm_judge），请设置这个 key
# export OPENAI_API_KEY=""

# 让脚本无论从哪里执行都能定位到 Parallel-Soft-Thinking 目录
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

# ===== 模型/Tokenizer/LoRA =====
# 基座模型（HF repo id 或本地路径）
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
# tokenizer（这里强制使用原版 tokenizer；如果不传，会默认用 lora_path 目录）
TOKENIZER_PATH="Qwen/Qwen2.5-7B-Instruct"
# 你的 LoRA checkpoint 目录（必须包含 lora_state.pt + lora_config.json），会自动检测peft类型和我们自己定义的类型
LORA_CKPT_DIR="/home/ubuntu/Documents/newdisk_22T/twq/soft_thinking_exp/Parallel-Soft-Thinking/models/checkpoints/soft_thinking_qwen2p5_7b/checkpoint-epoch-3-step-8958"

# ===== 运行资源配置 =====
TP_SIZE=2                    # tensor parallel 张数（--num_gpus）
MEM_FRACTION_STATIC=0.8      # 每张卡可用显存比例（--mem_fraction_static）
START_IDX=0                  # 数据集起始样本下标（--start_idx）
END_IDX=100000               # 数据集结束样本下标（--end_idx）
MAX_BATCH=1000000
NUM_SAMPLES=1                # 每个样本采样次数（--num_samples）

# ===== 生成/采样参数 =====
MAX_NEW_TOKENS=4096          # 最大生成 token 数（--max_generated_tokens）
TEMPERATURE=0.6              # 采样温度（--temperature）
TOP_P=0.95                   # nucleus sampling（--top_p）
TOP_K=30                     # top-k sampling（--top_k）
MIN_P=0.001                  # min-p 过滤（--min_p）
REPETITION_PENALTY=1.0       # 重复惩罚（--repetition_penalty）

# ===== Soft-thinking 参数 =====
ENABLE_SOFT_THINKING=true    # 是否开启 soft-thinking（--enable_soft_thinking）
MAX_TOPK=10                  # soft token 采用的 top-k 大小（--max_topk）
# soft-thinking 结束标记（对应 SamplingParams.think_end_str / 训练/提示模板里的 </think>）
THINK_END_STR="</think>"     # （--think_end_str）

# 进入softthinking的条件，开启后只有达到该条件才会触发soft_thinking
SOFT_THINK_TRIGGER_ENTROPY=0.4 # 设定一个大于0的值才开始softthinking
SOFT_THINKING_STEPS=3       # SOFTTHINKING步数
MAX_SOFT_THINKING_TRIGGERS=5    # 最大softthinking触发次数

# 退出/早停策略（在离散模式下用熵触发 early stop；soft-thinking 内也会用阈值控制退出）
EARLY_STOP_ENTROPY=0.01      # 熵阈值（--early_stopping_entropy_threshold，0 表示关闭）
EARLY_STOP_LEN=256           # 连续低熵步数门槛（--early_stopping_length_threshold）

# soft-thinking 前后阶段可以用不同的采样参数（after_thinking_*）
AFTER_T_TEMPERATURE=0.6      # 离开 soft-thinking 后的温度（--after_thinking_temperature）
AFTER_T_TOP_P=0.95           # 离开 soft-thinking 后的 top_p（--after_thinking_top_p）
AFTER_T_TOP_K=30             # 离开 soft-thinking 后的 top_k（--after_thinking_top_k）
AFTER_T_MIN_P=0.0            # 离开 soft-thinking 后的 min_p（--after_thinking_min_p）

# LoRA 应用模式：
# - joint：全程启用 LoRA（只要请求传了 lora_path）
# - split：仅在 soft-thinking 阶段启用 LoRA（prefill/离散 decode 都是 base-only）
LORA_MODE="split"            # （--lora_mode）

args=(
  # ===== 数据集/基础配置 =====
  --dataset "math500"                    # 评测数据集
  --model_name "${BASE_MODEL}"           # 推理基座模型
  --tokenizer_path "${TOKENIZER_PATH}"   # tokenizer 路径（强制原版）

  # ===== LoRA 相关 =====
  --lora_path "${LORA_CKPT_DIR}"         # LoRA checkpoint 目录（支持 lora_state.pt + lora_config.json）
  --lora_mode "${LORA_MODE}"             # split/joint

  # ===== soft-thinking 开关与控制 =====
  --max_topk "${MAX_TOPK}"               # soft token top-k 大小
#   --think_end_str "${THINK_END_STR}"     # soft-thinking 结束 token 字符串
  --enable_soft_thinking                 # 开启 soft-thinking 推理
  --soft_thinking_trigger_entropy "${SOFT_THINK_TRIGGER_ENTROPY}"
  --soft_thinking_steps "${SOFT_THINKING_STEPS}"
  --max_soft_thinking_triggers "${MAX_SOFT_THINKING_TRIGGERS}"

  # ===== 生成上限与采样 =====
  --max_generated_tokens "${MAX_NEW_TOKENS}"      # 最大生成 token 数
  --temperature "${TEMPERATURE}"                  # 温度
  --top_p "${TOP_P}"                              # top_p
  --top_k "${TOP_K}"                              # top_k
  --min_p "${MIN_P}"                              # min_p
  --repetition_penalty "${REPETITION_PENALTY}"    # 重复惩罚

  # ===== soft-thinking 结束后采样参数 =====
  --after_thinking_temperature "${AFTER_T_TEMPERATURE}"
  --after_thinking_top_p "${AFTER_T_TOP_P}"
  --after_thinking_top_k "${AFTER_T_TOP_K}"
  --after_thinking_min_p "${AFTER_T_MIN_P}"

  # ===== 早停/退出策略 =====
  --early_stopping_entropy_threshold "${EARLY_STOP_ENTROPY}"
  --early_stopping_length_threshold "${EARLY_STOP_LEN}"

  # ===== 资源与并行 =====
  --mem_fraction_static "${MEM_FRACTION_STATIC}"  # 显存占用上限
  --num_gpus "${TP_SIZE}"                         # TP 张数

  # ===== 数据集切片与采样次数 =====
  --start_idx "${START_IDX}"
  --end_idx "${END_IDX}"
  --num_samples "${NUM_SAMPLES}"
  --max_batch "${MAX_BATCH}"
)

python run_sglang_softthinking.py "${args[@]}"
