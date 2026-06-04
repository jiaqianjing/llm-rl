# DPO 训练数据构建

> 以 GSM8K 数学推理任务为例，记录从数据生成到 chosen/rejected 标注的完整流程。

---

## 一、为什么需要单独构建数据

DPO 训练需要 **(prompt, chosen, rejected)** 三元组。GSM8K 原始数据集只有 `{question, answer}`，没有偏好对。

解决方案：用 SFT 模型（Qwen2.5-7B-Instruct）对每道题采样多条回答，用 reward 函数自动标注 chosen/rejected。

---

## 二、Prompt 格式：必须用 Chat Template

**错误做法（我们踩过的坑）：**
```python
# raw 格式，不是模型预训练时见过的格式
prompt = "Solve the following math problem... write your final answer after ####.\nProblem: {q}\nSolution:"
```

这会导致 DPO 训练后模型产生灾难性遗忘（catastrophic forgetting）：
- 模型在 raw 格式上被 DPO 优化了 1000 步
- 评估时用 chat template，模型不认识这种输入
- 结果：GSM8K 准确率从 81% 跌到 43%

**正确做法：**
```python
messages = [{"role": "user", "content": question}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

实际 token 序列：
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Janet has 3 apples...<|im_end|>
<|im_start|>assistant
```

**原则：训练数据的 prompt 格式必须和模型预训练/SFT 的格式一致。**

---

## 三、采样策略

### 初版：每 prompt 采 2 条（已废弃）

```python
output_ids = model.generate(..., num_return_sequences=2, temperature=0.8)
```

- 模型准确率 81%，两条都对的概率 ≈ 0.81² ≈ 66%
- **yield 率只有 7.5%**（2000 prompt 只得到 149 对）

### 改进版：每 prompt 采 4 条，取最好/最差

```python
# 分两次调用以避免 KV cache OOM
for _ in range(2):
    output_ids = model.generate(..., num_return_sequences=2, temperature=0.8)
    responses += decode(output_ids)

rewards = [math_reward(r, answer) for r in responses]
chosen   = responses[rewards.index(max(rewards))]
rejected = responses[找最后一个 min(rewards)]
```

- 4 条全对的概率 ≈ 0.81⁴ ≈ 43%，有效对的概率更高
- **yield 率提升到 13%**（2000 prompt 得到 263 对）
- `num_return_sequences=4` 直接放在一个 generate 里会 OOM，需拆成 2+2

---

## 四、Reward 函数

### 实现

```python
def math_reward(response: str, ground_truth: str) -> float:
    # 第一步：优先匹配 #### 格式（GSM8K 标准格式）
    m = re.search(r"####\s*([\d,\.\-]+)", response)
    if m:
        return 1.0 if m.group(1).replace(",", "") == ground_truth else 0.0

    # 第二步：fallback，取回答中最后一个数字
    numbers = re.findall(r"(-?\d{1,10}(?:\.\d+)?)", response)
    if numbers:
        return 1.0 if numbers[-1] == ground_truth else 0.0

    return 0.0
```

### fallback 的风险

"I think the answer might be around 10 or 15, probably 15" → 取 15，如果答案是 10 则误判。

更严格的做法：只接受 `\boxed{}` 或 `####` 格式，不做 fallback，避免随机数字被误识别为答案。

---

## 五、业界 Reward 方案对比

| 方案 | 原理 | 适用场景 | 代表工作 |
|------|------|---------|---------|
| **Outcome Reward**（我们用的）| 只验证最终答案对错 | 数学、代码等可验证任务 | DeepSeek-R1, WizardMath |
| **Process Reward Model (PRM)** | 对每个推理步骤打分 | 需要过程正确的任务 | Math-Shepherd, Let's Reward Step by Step |
| **LLM-as-Judge** | 用强 LLM 对两条回答做偏好比较 | 主观任务（指令跟随、对话质量）| UltraFeedback |
| **Human Preference** | 人工标注偏好 | 高质量、高成本场景 | InstructGPT, Anthropic HH |

**为什么数学任务用 Outcome Reward：**
答案是客观的，不需要 LLM 判断。PRM 更精确但需要步骤级标注，对初期实验而言 outcome reward 够用。

---

## 六、数据格式

每条训练数据：
```json
{
  "prompt": "<|im_start|>user\nJanet has 3 apples...<|im_end|>\n<|im_start|>assistant\n",
  "chosen": "She gave away 2, so she has 3-2=1 apple left.",
  "rejected": "3 plus 2 equals 5, so she has 5 apples."
}
```

- `prompt`：包含完整 chat template，到 `<|im_start|>assistant\n` 为止（generation prompt）
- `chosen`：模型自然输出的文本，reward=1.0 的那条
- `rejected`：模型自然输出的文本，reward=0.0 的那条

DPO trainer 计算 `log p(chosen | prompt)` 和 `log p(rejected | prompt)` 时，把 prompt 和 response 拼接起来做 forward pass，用 response mask 只对 response 部分的 token 求 log prob。

---

## 七、yield 率与数据量的权衡

| 策略 | 采样数 | yield 率 | 2000 prompt 产出 |
|------|--------|---------|----------------|
| 2 条取好坏（raw format） | 2 | ~30% | 609 对 |
| 2 条取好坏（chat template）| 2 | 7.5% | 149 对 |
| 4 条取最好/最差（chat template）| 4 | 13% | 263 对 |

chat template 下 yield 更低，因为模型更强（81% vs 43%），大多数 prompt 4 条都答对。

进一步提升 yield 的方向：
- 提高采样温度（temperature 0.8 → 1.0）增加多样性
- 采更多条（8条、16条）
- 专门选难题（模型容易出错的 prompt）
- 使用较弱的基础模型做采样（但 DPO 目标模型仍是强模型）
