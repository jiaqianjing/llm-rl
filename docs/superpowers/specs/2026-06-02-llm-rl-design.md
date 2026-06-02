# llm-rl 设计文档

**日期**：2026-06-02  
**目标**：裸写 GRPO 与 DPO，在同一 base model 上对比两种算法的训练动态与最终效果  
**Base model**：Qwen2.5-7B-Instruct  
**硬件**：4x A100 80G（GPU 4-7）  
**参考框架**：oxRL（作为 sanity check 对照，非主线）

---

## 一、目标与范围

### 目标
1. **理解原理**：手写 GRPO 核心逻辑（group sampling、advantage 计算、policy gradient），看清每一行代码的含义
2. **直观对比**：在相同 base model + 相同数据上，对比 GRPO 与 DPO 的训练曲线和最终效果
3. **两阶段推进**：Phase 1 用数学任务跑通机制，Phase 2 扩展到通用指令跟随做横向对比

### 不在范围内
- PPO（有 critic/value network，复杂度更高，后续可扩展）
- 异步分布式 rollout（串行足够，不引入 Ray）
- 量化训练（80G 显存充裕，不需要）

---

## 二、算法核心

### DPO

```
输入：(prompt, chosen, rejected) — 静态数据集

logr_chosen   = log π_θ(chosen|prompt)   - log π_ref(chosen|prompt)
logr_rejected = log π_θ(rejected|prompt) - log π_ref(rejected|prompt)

L_DPO = -log σ( β * (logr_chosen - logr_rejected) )
```

- **无生成**：训练时不采样新内容，梯度信号来自标注数据
- **β**：控制偏离 ref model 的幅度（默认 0.1）
- **ref model**：SFT checkpoint，整个训练冻结

### GRPO

```
输入：prompt only — 在线生成

responses = model.generate(prompt, n=G)          # G=8 条
rewards   = reward_fn(prompt, responses)          # [B, G]

advantages = (rewards - mean(rewards)) / (std(rewards) + ε)

L_pg  = -(advantages * log π_θ(responses|prompt)).mean()
L_kl  = (log π_θ - log π_ref).mean()
L     = L_pg + β * L_kl
```

- **在线探索**：每步生成新 response，不依赖标注
- **group-relative advantage**：无需 critic/value network（相比 PPO 的核心简化）
- **β**：KL 惩罚系数（默认 0.04）
- **"全错无信号"陷阱**：G 条 response 全错时 std=0，advantage 全为 0，无梯度——需要课程学习或 reward shaping 应对

### Ref model 说明

| 角色 | 更新策略 | 用途 |
|---|---|---|
| KL 锚点（π_ref） | 全程冻结 | 防止偏离 SFT 起点，DPO/GRPO 共用 |
| IS 分母（π_old） | 每步同步 | PPO 专用，GRPO 不需要 |

---

## 三、项目结构

```
llm-rl/
├── data/
│   ├── gsm8k.py              # {prompt, answer}
│   ├── math_hard.py          # Phase 2 扩展
│   └── ultrafeedback.py      # {prompt, chosen, rejected}
├── rewards/
│   ├── math_reward.py        # exact match → 0.0 / 1.0
│   ├── format_reward.py      # 格式奖励（可选）
│   └── llm_judge.py          # LLM-as-judge（UltraFeedback GRPO）
├── algorithms/
│   ├── grpo.py               # group advantage + policy gradient loss
│   └── dpo.py                # log ratio contrastive loss
├── rollout/
│   └── vllm_rollout.py       # Phase 2 vLLM rollout
├── trainer/
│   ├── base_trainer.py       # 模型加载、FSDP、checkpoint、W&B
│   ├── grpo_trainer.py       # GRPO 训练循环
│   └── dpo_trainer.py        # DPO 训练循环
├── eval/
│   ├── gsm8k_eval.py         # test set accuracy
│   └── win_rate.py           # LLM judge win rate（Phase 2）
├── configs/
│   ├── fsdp_4gpu.yaml
│   ├── grpo_gsm8k.yaml
│   ├── dpo_gsm8k.yaml
│   └── grpo_ultrafeedback.yaml
├── scripts/
│   └── plot_curves.py        # 从 W&B 拉数据画对比曲线
├── train_grpo.py
├── train_dpo.py
└── docs/
    ├── grpo-vs-dpo-concepts.md
    ├── rl-insights-and-meta-learning.md
    ├── data-pipeline-and-distributed.md
    └── superpowers/specs/2026-06-02-llm-rl-design.md
```

---

## 四、数据方案

### Phase 1：数学推理（GSM8K）

| 算法 | 数据准备 |
|---|---|
| GRPO | 直接用 prompt，无需标注 |
| DPO | 用 SFT 模型对每条 prompt 采样 2 条 response，math_reward 打标，reward 不同的对构成 (chosen, rejected)，两者相同则丢弃 |

### Phase 2：指令跟随（UltraFeedback）

| 算法 | 数据准备 |
|---|---|
| GRPO | 只用 prompt，LLM judge 打分作 reward |
| DPO | 直接使用 (prompt, chosen, rejected) |

---

## 五、分布式方案

**内存估算（Qwen2.5-7B，bf16，4x A100 80G）：**

```
FSDP ZeRO-3 训练模型：  ~14 GB/卡（分片后 ~3.5GB 权重 + 梯度 + 优化器）
参考模型（全量复制）：  ~14 GB/卡
激活 + micro-batch：    ~10 GB/卡
─────────────────────────────────────
合计：                  ~38 GB/卡  ✓
```

**Phase 1（model.generate() rollout）：**
- FSDP ZeRO-3 + ref model 全量复制，全在 GPU 4-7
- rollout 用 `model.generate()`，无额外进程

**Phase 2（vLLM rollout）：**
- rollout 阶段：vLLM tensor parallel 4 卡
- 训练阶段：Accelerate FSDP ZeRO-3
- 两阶段串行，每步后 `vllm_engine.update_weights(state_dict)`

---

## 六、实验设计

### 对比实验矩阵

| Run | 算法 | 数据 | 目的 |
|---|---|---|---|
| A | GRPO | GSM8K | Phase 1 主实验 |
| B | DPO | GSM8K preference | Phase 1 对照 |
| C | GRPO | UltraFeedback | Phase 2 主实验 |
| D | DPO | UltraFeedback | Phase 2 对照 |

### 追踪指标

**GRPO 专属：**
- `reward/mean`：每步平均 reward（期望上升）
- `reward/pass_rate`：reward=1 的比例
- `reward/std`：组内方差（中期高，后期收窄）

**DPO 专属：**
- `logr/chosen`：期望上升
- `logr/rejected`：期望下降
- `reward_margin`：两者之差，期望扩大

**共同：**
- `loss/train`、`kl/mean`、`eval/accuracy`、`lr`

### 关键超参

```yaml
# GRPO
group_size: 8
beta: 0.04
lr: 1e-6
batch_size: 4          # 实际生成 32 条/step

# DPO
beta: 0.1
lr: 5e-7
batch_size: 16
```

### 预期观察

```
GRPO：reward/mean 从 ~0.2 爬升到 0.6+
      reward/std 先宽后窄（探索 → 收敛）

DPO： reward_margin 单调上升
      eval/accuracy 提升幅度通常比 GRPO 小
      （受限于 preference pair 质量和覆盖度）
```

---

## 七、实现路线

```
Week 1：base_trainer + DPO（离线，无生成，最快跑通）
Week 2：GRPO Phase 1（model.generate() rollout + math reward）
Week 3：对比实验 A vs B，分析曲线差异
Week 4：Phase 2 扩展（UltraFeedback + vLLM rollout）
```

---

## 八、oxRL 对照计划

用 oxRL 跑相同的 GSM8K 实验（task=math），对比：
- loss 曲线形状是否一致
- 最终 accuracy 是否接近
- 找出自己实现与工业级框架的差距

oxRL 命令：
```bash
oxrl train --model Qwen/Qwen2.5-7B-Instruct \
           --task math \
           --dataset gsm8k \
           --epochs 3
```
