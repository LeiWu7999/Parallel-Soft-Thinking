# python ./models/download.py --model_name "Qwen/QwQ-32B"

export OPENAI_API_KEY=""

python run_sglang_softthinking.py \
    --dataset "math500" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --max_topk 10 \
    --max_generated_tokens 4096 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --after_thinking_temperature 0.6 \
    --after_thinking_top_p 0.95 \
    --after_thinking_top_k 30 \
    --after_thinking_min_p 0.0 \
    --early_stopping_entropy_threshold 0.01 \
    --early_stopping_length_threshold 256 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 100000 \
    --num_gpus 2 \
    --num_samples 1 \
    --enable_soft_thinking \
