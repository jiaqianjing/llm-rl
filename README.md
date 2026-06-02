# llm-rl

从零手写 GRPO 与 DPO，在 Qwen2.5-7B-Instruct 上对比两种算法的训练动态与效果。

## 目标

- 裸写 GRPO 核心逻辑：group sampling、advantage 计算、policy gradient
- 裸写 DPO 核心逻辑：log ratio contrastive loss
- Phase 1：GSM8K 数学推理（verifiable reward）
- Phase 2：UltraFeedback 指令跟随（LLM judge reward）
- 用 [oxRL](https://github.com/warlockee/oxRL) 做 sanity check 对照

## 算法对比

| | DPO | GRPO |
|---|---|---|
| **数据** | 静态 (prompt, chosen, rejected) | 只需 prompt，运行时在线生成 |
| **Reward** | 隐式（藏在偏好对里）| 显式标量函数（答对=1，答错=0）|
| **探索** | 无，分布固定于训练集 | 有，每步采样新 response |
| **计算量** | 低（纯 forward pass）| 高（rollout 生成 + forward）|
| **适合任务** | 主观偏好（指令跟随、对话风格）| 可验证任务（数学、代码、工具调用）|

## 代码结构

```
algorithms/          # 纯 PyTorch 数学内核，不加载模型
  grpo.py            # compute_advantages + grpo_loss
  dpo.py             # dpo_loss
  utils.py           # sequence_logprobs（policy 与 ref 共用）

trainer/
  base_trainer.py    # 加载 policy 模型 + 冻结 ref 模型，Accelerate + W&B
  grpo_trainer.py    # rollout → reward → logprob → GRPO loss
  dpo_trainer.py     # tokenize pairs → logprob → DPO loss

data/
  gsm8k.py           # 加载 HuggingFace GSM8K，格式化为 {prompt, answer}

rewards/
  math_reward.py     # 提取 #### 后数字，精确匹配 → 0/1 reward

eval/
  gsm8k_eval.py      # 贪心解码评测 GSM8K test set accuracy

scripts/
  generate_gsm8k_pairs.py   # 用 SFT 模型采样生成 DPO preference pairs
  smoke_test.py             # 3-step 端到端冒烟测试（tiny-gpt2，无需 GPU）

configs/
  grpo_gsm8k.yaml    # GRPO 超参（β=0.04，G=8，lr=1e-6）
  dpo_gsm8k.yaml     # DPO 超参（β=0.1，lr=5e-7）
  fsdp_4gpu.yaml     # Accelerate FSDP ZeRO-3，4 卡，bf16
```

## 快速开始

**安装依赖：**

```bash
pip install -r requirements.txt
```

**单元测试（无 GPU）：**

```bash
pytest
```

**端到端冒烟测试（tiny-gpt2，CPU，~30s）：**

```bash
python scripts/smoke_test.py
```

**生成 DPO preference 数据（需要 GPU + Qwen2.5-7B-Instruct）：**

```bash
python scripts/generate_gsm8k_pairs.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output data/gsm8k_preference_train.json \
  --num_samples 2000
```

**训练（4x A100，GPU 4-7）：**

```bash
# GRPO on GSM8K
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --config_file configs/fsdp_4gpu.yaml \
  train_grpo.py --config configs/grpo_gsm8k.yaml

# DPO on GSM8K（需先生成 preference 数据）
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --config_file configs/fsdp_4gpu.yaml \
  train_dpo.py --config configs/dpo_gsm8k.yaml
```

**评测 checkpoint：**

```bash
python eval/gsm8k_eval.py --model checkpoints/grpo-gsm8k/step-200
```

## 关键实现细节

**GRPO rollout**：`model.generate(num_return_sequences=G)` 对每个 prompt 采样 G 条 response，组内归一化 advantage，policy gradient + KL 惩罚。Phase 1 直接用训练模型生成（权重天然同步），Phase 2 接入 vLLM 提速。

**Ref model**：全量加载、全程冻结，不走 FSDP（只移到 device）。DPO 和 GRPO 都保留一份用于计算 KL / log ratio。

**sequence_logprobs**：`logits[:, :-1]` vs `input_ids[:, 1:]` 做 shift，`response_mask` 只对 response 部分 token 求和，prompt 部分不计入 loss。

**内存估算（Qwen2.5-7B，bf16，4 卡）**：分片权重 ~3.5GB + 梯度 ~3.5GB + Adam states ~7GB + ref model ~14GB + 激活 ~10GB ≈ 38GB/卡，80GB A100 绰绰有余。

## 硬件

4x A100 80G（GPU 4-7），Accelerate FSDP ZeRO-3，bf16

## 文档

- [GRPO vs DPO 概念详解](docs/grpo-vs-dpo-concepts.md)
- [数据流与分布式方案](docs/data-pipeline-and-distributed.md)
- [RL 本质与 Meta-Learning](docs/rl-insights-and-meta-learning.md)

## 参考

- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948)（GRPO 来源）
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [oxRL](https://github.com/warlockee/oxRL)（对照框架）
