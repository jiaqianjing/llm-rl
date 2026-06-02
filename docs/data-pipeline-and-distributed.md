# 数据流与分布式训练方案

---

## 一、数据计划（两个阶段）

| 阶段 | 任务 | 数据集 | GRPO 用法 | DPO 用法 |
|---|---|---|---|---|
| Phase 1 | 数学推理 | GSM8K (7473 条) | 只用 prompt，在线生成 | 用 SFT 模型生成 chosen/rejected，reward 打标 |
| Phase 2 | 指令跟随 | UltraFeedback (60k 条) | 只用 prompt，LLM judge 打分 | 直接用现成的 (prompt, chosen, rejected) |

GSM8K reward 函数：

```python
import re

def math_reward(response: str, ground_truth: str) -> float:
    # 提取 #### 后面的数字（GSM8K 标准格式）
    match = re.search(r"####\s*([\d\.\-]+)", response)
    if not match:
        return 0.0
    return 1.0 if match.group(1).strip() == ground_truth.strip() else 0.0
```

---

## 二、GRPO 数据流

```
每个训练 step：

  ① 取一个 batch 的 prompts（无需 label）
           │
  ② model.generate(prompts, num_return_sequences=G)
           │  G=8 条 response/prompt
  ③ reward_fn(prompts, responses)  →  rewards [B, G]
           │
  ④ 组内归一化  →  advantages [B, G]
           │
  ⑤ 训练模型 forward  →  log_probs [B, G]
     参考模型 forward  →  ref_log_probs [B, G]  (no_grad)
           │
  ⑥ L = -(advantages * log_probs).mean()
       + β * (log_probs - ref_log_probs).mean()
           │
  ⑦ backward + optimizer.step()
```

---

## 三、DPO 数据流

```
每个训练 step：

  ① 取一个 batch 的 (prompt, chosen, rejected)
           │
  ② 训练模型 forward（chosen + rejected 拼成一个 batch 一次过）
     → log_pi_chosen, log_pi_rejected
           │
  ③ 参考模型 forward（no_grad）
     → log_ref_chosen, log_ref_rejected
           │
  ④ logr_chosen   = log_pi_chosen   - log_ref_chosen
     logr_rejected = log_pi_rejected - log_ref_rejected
           │
  ⑤ L = -log_sigmoid(β * (logr_chosen - logr_rejected)).mean()
           │
  ⑥ backward + optimizer.step()
```

> **效率注意**：DPO 的 step ② 可以把 chosen 和 rejected 拼在同一个 batch 里一次 forward，
> 效率比 GRPO 高很多（GRPO 需要额外的 rollout 生成开销）。

---

## 四、分布式方案（4x A100 80G，GPU 4-7）

### 内存估算（Qwen2.5-7B，bf16）

```
模型权重（FSDP ZeRO-3 分片）：14GB / 4 ≈  3.5 GB/卡
梯度：                                  ≈  3.5 GB/卡
AdamW optimizer states：                ≈  7.0 GB/卡
参考模型（全量复制，frozen）：           ≈ 14.0 GB/卡
激活值 + micro-batch：                  ≈ 10.0 GB/卡
─────────────────────────────────────────────────────
合计：                                  ≈ 38 GB/卡   ✓ 80GB 绰绰有余
```

### Phase 1：`model.generate()` rollout

```
GPU 4-7
  ├── 训练模型（FSDP ZeRO-3，权重分片）
  ├── 参考模型（全量复制，no_grad，frozen）
  └── rollout 直接调 model.generate()，无需额外进程
```

简单、无进程间通信、权重天然同步，适合理解原理阶段。

### Phase 2：vLLM rollout（提速阶段）

```
Rollout 阶段（串行）：
  GPU 4-7  →  vLLM tensor parallel 4 卡，批量生成 G 条 response

训练阶段：
  GPU 4-7  →  Accelerate FSDP ZeRO-3，计算 loss + backward

两阶段串行交替，每步结束后同步权重：
  state_dict = accelerator.get_state_dict(model)
  vllm_engine.update_weights(state_dict)
```

串行（先 rollout 再 train）比异步并发简单得多，4x A100 推理速度足够，无需流水线。

---

## 五、启动命令

```bash
# GRPO 训练（GPU 4-7，4 卡 FSDP）
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --config_file configs/fsdp_4gpu.yaml \
  train_grpo.py \
  --config configs/grpo_gsm8k.yaml

# DPO 训练
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --config_file configs/fsdp_4gpu.yaml \
  train_dpo.py \
  --config configs/dpo_gsm8k.yaml
```

---

## 六、两种算法数据流对比

| | GRPO | DPO |
|---|---|---|
| **数据输入** | prompt only | (prompt, chosen, rejected) |
| **生成开销** | 每步 G×B 次生成 | 无生成 |
| **forward 次数/step** | 2（训练模型 + ref）+ rollout | 2（训练模型 + ref）|
| **瓶颈** | rollout 生成速度 | forward pass 速度 |
| **GPU 利用率** | rollout 阶段利用率低 | 全程高 |
