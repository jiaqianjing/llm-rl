# GRPO vs DPO：核心概念与 Loss 详解

> 面向已掌握 SFT + DPO 的读者，重点梳理 GRPO 的原理以及两者的本质差异。

---

## 一、两种算法的哲学差异

| | DPO | GRPO |
|---|---|---|
| **数据** | 静态 (prompt, chosen, rejected) | 只需 prompt，运行时在线生成 |
| **Reward** | 隐式（藏在偏好对里） | 显式标量函数（答对=1，答错=0）|
| **探索** | 无，分布固定于训练集 | 有，每步采样新 response |
| **优势估计** | 不需要 | 组内均值归一化（无需 critic）|
| **计算量** | 低（纯 forward pass） | 高（rollout 生成 + forward）|
| **适合任务** | 主观偏好（指令跟随、对话风格）| 可验证任务（数学、代码、工具调用）|
| **天花板** | 受限于标注数据质量 | 受限于 reward 函数质量 |

**一句话概括：**
- DPO："在这两个固定答案里，调整相对偏好"——静态对比，不生成，梯度信号来自标注数据
- GRPO："你刚才生成的 G 个答案，按组内排名给梯度"——动态探索，边生成边学，梯度信号来自 reward 函数

---

## 二、DPO Loss 详解

### 公式

```
L_DPO = -log σ( β * ( logr_chosen - logr_rejected ) )

logr = log π_θ(y|x) - log π_ref(y|x)   # 相对于参考策略的 log ratio
```

### 具体数值例子

prompt：`"Solve: 2x + 3 = 7"`

| | chosen `"x = 2"` | rejected `"x = 5"` |
|---|---|---|
| log π_θ（当前模型） | -0.5 | -1.2 |
| log π_ref（SFT 模型） | -0.8 | -1.0 |
| logr = θ - ref | **+0.3** | **-0.2** |

```
margin = β * (logr_chosen - logr_rejected)
       = 0.1 * (0.3 - (-0.2)) = 0.05

L_DPO = -log σ(0.05) ≈ 0.687   # loss 还很高，模型需要继续训练
```

### 梯度方向

- 把 `logr_chosen` 推高：让模型比 ref 更倾向于 chosen
- 把 `logr_rejected` 推低：让模型比 ref 更远离 rejected

DPO 不是让模型直接最大化 chosen 的绝对概率，而是最大化**相对于 ref 的概率比值之差**。β 控制偏离 ref 的幅度。

---

## 三、GRPO Loss 详解

### 公式

```
# Step 1: 生成 G 条 response 并打分
rewards ∈ R^[B, G]

# Step 2: 组内归一化 advantage
A_ij = (r_ij - mean(r_i)) / (std(r_i) + ε)

# Step 3: Policy gradient loss
L_pg = -mean( A_ij * log π_θ(response_ij | prompt_i) )

# Step 4: KL 惩罚
KL_ij = Σ_token [ log π_θ(t) - log π_ref(t) ]

# Total
L = L_pg + β * mean(KL)
```

### 具体数值例子

prompt：`"Solve: 2x + 3 = 7"`，group size G = 4

**生成与打分：**

| 编号 | response | reward |
|---|---|---|
| r1 | `"2x=4, x=2 ✓"` | 1.0 |
| r2 | `"2x=10, x=5 ✗"` | 0.0 |
| r3 | `"x=2 ✓"` | 1.0 |
| r4 | `"2x=4, x=5 ✗"` | 0.0 |

**组内归一化：**

```
mean_r = 0.5,  std_r = 0.5

A1 = (1.0 - 0.5) / 0.5 = +1.0   # 比平均好，强化
A2 = (0.0 - 0.5) / 0.5 = -1.0   # 比平均差，抑制
A3 = +1.0
A4 = -1.0
```

**Policy gradient loss（假设各 response 的 log prob）：**

```
L_pg = -( (+1.0)*(-0.3) + (-1.0)*(-1.5) + (+1.0)*(-0.4) + (-1.0)*(-1.8) ) / 4
     = -( -0.3 + 1.5 - 0.4 + 1.8 ) / 4
     = -0.65
```

最小化 L_pg → 推高 r1/r3 的生成概率，推低 r2/r4 的生成概率。

### 重要边界情况

| 情形 | rewards | std | advantage | 梯度 |
|---|---|---|---|---|
| 4 个全对 | [1,1,1,1] | 0 | 全为 0 | **无信号**（已经很好了）|
| 4 个全错 | [0,0,0,0] | 0 | 全为 0 | **无信号**（需要课程学习或 reward shaping）|
| 混合 | [1,0,1,0] | 0.5 | ±1.0 | 有效梯度 |

"全错无信号"是 GRPO 初期训练的常见陷阱：base model 太弱时模型无法自举。

---

## 四、Reference Model 详解

强化学习中 ref model 有**两个不同角色**，常被混淆：

### 角色 1：KL 锚点（固定不动）

```
KL( π_current || π_ref )    # π_ref = 初始 SFT checkpoint，整个训练冻结
```

- DPO、GRPO 默认用这种
- 防止模型偏离 SFT 起点太远（catastrophic forgetting）

### 角色 2：重要性采样分母（滚动更新）

PPO 的经典设计：

```
ratio = π_current(a|s) / π_old(a|s)    # clip 到 [1-ε, 1+ε]
```

- `π_old` 每隔几步同步为当前策略，允许同一批 rollout 数据做多次梯度更新
- 这是 PPO 比 GRPO 复杂的地方之一

### 对照表

| | π_ref（KL 锚点）| π_old（IS 分母）|
|---|---|---|
| **用途** | 防偏移，正则化 | 允许多步复用 rollout 数据 |
| **更新频率** | 固定（整个训练） | 每轮 rollout 后同步 |
| **GRPO** | 有 | 无（每步更新，ratio≈1，跳过 clip）|
| **PPO** | 有 | 有（核心机制）|
| **DPO** | 有 | 无（离线，不涉及采样）|

**GRPO 相比 PPO 的简化**：去掉了 π_old 这个滚动 ref，每步 rollout 完立即更新，不复用数据，实现更简单。

---

## 五、vLLM Online Rollout 的权重同步问题

GRPO 需要用**当前策略 π_θ** 做 rollout，但 vLLM 和训练框架是独立进程，权重各存一份。

### 三种同步策略

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 每步同步 | 训练完一步 → push 新权重给 vLLM | 最正确，无 staleness | 每步多一次权重拷贝 |
| 每 N 步同步 | 每 N 步同步一次，中间用旧权重 | 减少同步开销 | 有 staleness，需 importance sampling 修正 |
| `model.generate()` | 不用 vLLM，直接用训练模型生成 | 零工程负担，天然同步 | 生成速度慢 3-5 倍 |

### 本项目策略

```
Phase 1（理解原理）：model.generate() 做 rollout，4 卡 FSDP 训练
Phase 2（提速）：    vLLM tensor parallel + 每步权重热更新
```

vLLM 0.4+ 支持 `update_weights()` 热更新，无需重启引擎：

```python
# 训练完一步后同步权重
state_dict = accelerator.get_state_dict(model)
vllm_engine.update_weights(state_dict)
```

---

## 六、小结：选哪个算法？

```
有可验证的 reward 函数（数学、代码）  →  用 GRPO
只有人类偏好数据                      →  用 DPO
想要最简单的实现                       →  用 DPO
想要模型自我探索、突破标注上限          →  用 GRPO
```
