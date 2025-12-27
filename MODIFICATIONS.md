# Parallel Soft Thinking - 核心修改说明

本文档说明了基于 SGLang Soft Thinking 版本所做的核心修改与增强。

所有新增或修改的代码均以如下标记清晰标注：
```python
# ==========
# begin of parallel soft thinking
# ==========
```
和
```python
# ==========
# end of parallel soft thinking
# ==========
```

---

## 1. 动态熵触发机制

在原有静态模式基础上，新增动态触发控制：

| 参数 | 说明 |
|:---|:---|
| `soft_thinking_trigger_entropy` | 熵阈值触发 |
| `random_think_prob` | 随机触发概率 |
| `max_soft_thinking_triggers` | 单请求最大触发次数 |

**核心代码** (`schedule_batch.py`):

```python
        # Dynamic Soft Thinking Logic
        # 1) 熵触发：soft_thinking_trigger_entropy > 0
        # 2) 随机触发：random_think_prob > 0
        dynamic_entropy_enabled = (
            self.sampling_params.soft_thinking_trigger_entropy > 0
        )
        random_enabled = getattr(self.sampling_params, "random_think_prob", 0.0) > 0.0
        dynamic_enabled = dynamic_entropy_enabled or random_enabled

        # 触发条件检查
        if (not suppress_trigger) and dynamic_enabled and (
            self.trigger_count < self.sampling_params.max_soft_thinking_triggers
        ):
            # 1) 熵触发策略
            if dynamic_entropy_enabled and (
                self.entropy >= self.sampling_params.soft_thinking_trigger_entropy
            ):
                triggered = True

            # 2) 随机触发策略
            if (not triggered) and random_enabled:
                if random.random() < self.sampling_params.random_think_prob:
                    triggered = True

            if triggered:
                self.sampling_params.soft_thinking_mode = True
                self.trigger_count += 1
                # Record entry bookkeeping for KV masking / position rollback
                self.st_entry_pos = cur_pos
                self.st_entry_token_id = self.output_ids[-1]
                self.st_entry_entropy = float(self.entropy)
```

---

## 2. KV Cache Masking 机制

**核心改进**: 采用逻辑屏蔽替代物理删除，兼容 RadixCache。

通过 `st_mask_ranges` 记录需要屏蔽的区间：

```python
            # Each element is an inclusive (start_pos, end_pos).
            self.st_mask_ranges: List[Tuple[int, int]] = []
```

**区间记录** (`scheduler_output_processor_mixin.py`):

```python
                if finalize_st_mask:
                    entry_pos = getattr(req, "st_entry_pos", None)
                    soft_end_pos = getattr(req, "st_soft_end_pos", None)
                    if (
                        entry_pos is not None
                        and soft_end_pos is not None
                        and entry_pos <= soft_end_pos
                    ):
                        req.st_mask_ranges.append((entry_pos, soft_end_pos))
```

---

## 3. Entropy-Compare Decoding

基于熵收益的决策机制，选择更低熵（更确信）的分布：

```python
                # Entropy-compare decoding on the insertion-forward step:
                # compare the original entry-step logits entropy vs the insertion-forward entropy,
                # and decode using the lower-entropy distribution.
                if finalize_st_mask:
                    entry_entropy = getattr(req, "st_entry_entropy", None)
                    entry_logits = getattr(req, "st_entry_full_logits", None)
                    ins_entropy = float(logits_output.entropy[i])
                    ins_logits = logits_output.next_token_logits[i].detach().cpu().clone()

                    # Decide which distribution to trust (prefer lower entropy)
                    use_entry = False
                    if entry_logits is not None and entry_entropy is not None:
                        if ins_entropy is None or ins_logits is None:
                            use_entry = True
                        elif entry_entropy <= ins_entropy:
                            use_entry = True

                    chosen_logits = entry_logits if use_entry else ins_logits
                    final_token_id = int(torch.multinomial(probs, num_samples=1).item())
```

**Token 替换与位置回滚**:

```python
                # When soft thinking ends, replace the next discrete token by the
                # entry discrete token, and roll back position encoding.
                if getattr(req, "st_pending_replace", False):
                    entry_token_id = getattr(req, "st_entry_token_id", None)
                    entry_pos = getattr(req, "st_entry_pos", None)
                    soft_end_pos = getattr(req, "st_soft_end_pos", None)
                    if entry_token_id is not None and entry_pos is not None:
                        final_token_id = int(entry_token_id)
                        # 处理位置编码偏移
                        req.pos_offset += (soft_end_pos - entry_pos) + 1
                        req.st_pending_replace = False
                        req.st_pending_insertion_forward = True
```

---

## 4. 状态机管理

请求级别的状态追踪字段：

```python
            # Entry position and token for KV masking and position rollback
            self.st_entry_pos: Optional[int] = None
            self.st_entry_token_id: Optional[int] = None
            self.st_soft_end_pos: Optional[int] = None
            
            # Entry-step logits snapshot for entropy-compare decoding
            self.st_entry_entropy: Optional[float] = None
            self.st_entry_full_logits: Optional[torch.Tensor] = None
            
            # Pending flags for two-phase commit
            self.st_pending_replace: bool = False
            self.st_pending_insertion_forward: bool = False
            self.st_inflight_insertion_forward: bool = False
            
            # Accumulated mask ranges
            self.st_mask_ranges: List[Tuple[int, int]] = []
```

---

## 5. LoRA 阶段级控制

支持 `lora_mode` 参数控制 LoRA 的应用时机：

| 模式 | 说明 |
|:---|:---|
| `joint` | 始终应用 LoRA（默认） |
| `split` | 仅在 Soft Thinking decode 阶段启用 LoRA |

**配置定义** (`server_args.py`):

```python
    # How LoRA is applied during inference.
    # - "joint": always apply LoRA when lora_path is provided.
    # - "split": apply LoRA only during soft-thinking decode steps.
    lora_mode: str = "joint"
```

**核心路由逻辑** (`schedule_batch.py`):

```python
        # LoRA routing
        # - joint: always use req.lora_path
        # - split: only enable LoRA during soft-thinking steps. Prefill/extend
        #          always uses base (None). For mixed chunked prefill, only the
        #          decoding part can enable LoRA.
        lora_mode = global_server_args_dict.get("lora_mode", "joint")
        
        def _is_insertion_step(req: Req) -> bool:
            return bool(
                getattr(req, "st_pending_replace", False)
                or getattr(req, "st_pending_insertion_forward", False)
                or getattr(req, "st_inflight_insertion_forward", False)
            )
        
        if lora_mode == "split" and self.enable_soft_thinking:
            if self.forward_mode.is_extend():
                # Prefill 阶段不启用 LoRA
                if self.forward_mode.is_mixed() and self.decoding_reqs is not None:
                    # Mixed chunked prefill: 只有 decode 部分可以启用 LoRA
                    decoding_req_set = set(self.decoding_reqs)
                    lora_paths = [
                        (
                            req.lora_path
                            if (req in decoding_req_set)
                            and req.sampling_params.soft_thinking_mode
                            and not _is_insertion_step(req)
                            else None
                        )
                        for req in self.reqs
                    ]
                else:
                    lora_paths = [None] * len(self.reqs)
            else:
                # Decode 阶段：仅 soft_thinking_mode 且非 insertion step 时启用
                lora_paths = [
                    req.lora_path
                    if (
                        req.sampling_params.soft_thinking_mode
                        and not _is_insertion_step(req)
                    )
                    else None
                    for req in self.reqs
                ]
```

---

## 6. 位置编码回滚机制

Soft Thinking 结束后，需要回滚位置编码以保持与 KV Cache Masking 的一致性。

**偏移量记录** (`schedule_batch.py`):

```python
            # Position encoding rollback accumulator (physical_pos - logical_pos).
            self.pos_offset: int = 0
```

**偏移量计算** (退出 Soft Thinking 时):

```python
                # When soft thinking ends, roll back position encoding
                if entry_token_id is not None and entry_pos is not None:
                    # 偏移量 = (soft_end_pos - entry_pos) + 1
                    req.pos_offset += (soft_end_pos - entry_pos) + 1
```

**位置编码应用** (`forward_batch_info.py`):

```python
                # 偏移位置编码
                positions = clamp_position(batch.seq_lens)
                if getattr(batch, "pos_offsets", None) is not None:
                    pos_offsets = batch.pos_offsets.to(device, non_blocking=True)
                    positions = torch.clamp(positions - pos_offsets, min=0)
                ret.positions = positions
```

**工作原理**:
- 物理位置 (physical_pos): KV Cache 中的实际位置
- 逻辑位置 (logical_pos): 模型看到的位置编码
- `pos_offset = physical_pos - logical_pos`
- 后续 Token 的位置编码 = `seq_len - pos_offset`，实现"跳过" Soft Thinking 区间的效果

---

## 7. 关键文件索引

| 文件 | 核心功能 |
|:---|:---|
| `managers/schedule_batch.py` | 动态触发、状态管理、LoRA 路由 |
| `managers/scheduler_output_processor_mixin.py` | 熵比较、Token 替换、Mask 记录 |
| `model_executor/forward_batch_info.py` | 位置编码偏移应用 |
| `layers/sampler.py` | 采样与 Top-K/P 过滤 |
| `layers/vocab_parallel_embedding.py` | Soft Token 加权 Embedding |
| `models/qwen2.py` | 模型 Forward 集成 |
| `server_args.py` | 服务参数定义 |

---

## 8. 与原版主要差异

| 维度 | 原版 Soft Thinking | Parallel Soft Thinking |
|:---|:---|:---|
| KV Cache 回退 | 物理删除 | Attention Masking |
| 触发机制 | 静态模式 | 动态熵触发 + 随机触发 |
| 位置编码 | 无回滚 | pos_offset 偏移回滚 |
| 并发支持 | 单 Batch | Continuous Batching |
| LoRA Gate | Token 级 | 阶段级 (split 模式) |

---

## Quick Start

脚本运行
```bash
bash scripts/st/qwen2_5_7b_st_lora_math.sh
```

```bash
python -m sglang.launch_server --model-path <your_model> \
    --soft-thinking-enabled \
    --soft-thinking-trigger-entropy 1.5 \
    --max-soft-thinking-triggers 10 \
    --soft-thinking-steps 5 \
    --lora-mode split
```

