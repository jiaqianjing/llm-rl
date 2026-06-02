# llm-rl Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Qwen2.5-7B-Instruct 上裸写 GRPO 与 DPO，在 GSM8K 数学任务上对比两种算法的训练动态，所有核心逻辑手写，不依赖 TRL。

**Architecture:** DPO 和 GRPO 共享 base_trainer（模型加载、FSDP、ref model、W&B），仅在 algorithms/ 层分叉。GRPO Phase 1 用 `model.generate()` 做 rollout（无 vLLM），DPO 用预先生成的 GSM8K preference pairs。

**Tech Stack:** Python 3.10+, PyTorch 2.0+, transformers 4.40+, accelerate 0.30+, datasets, wandb, pytest

---

## 文件结构

```
llm-rl/
├── algorithms/
│   ├── __init__.py          # 新建
│   ├── utils.py             # 新建：sequence_logprobs 工具函数
│   ├── dpo.py               # 新建：DPO loss
│   └── grpo.py              # 新建：GRPO loss + advantage 计算
├── data/
│   ├── __init__.py          # 新建
│   └── gsm8k.py             # 新建：加载 GSM8K，生成 preference pairs
├── rewards/
│   ├── __init__.py          # 新建
│   └── math_reward.py       # 新建：答案提取 + exact match
├── trainer/
│   ├── __init__.py          # 新建
│   ├── base_trainer.py      # 新建：模型加载、FSDP、ref model、W&B、checkpoint
│   ├── dpo_trainer.py       # 新建：DPO 训练循环
│   └── grpo_trainer.py      # 新建：GRPO 训练循环（model.generate() rollout）
├── eval/
│   ├── __init__.py          # 新建
│   └── gsm8k_eval.py        # 新建：test set accuracy 评估
├── configs/
│   ├── fsdp_4gpu.yaml       # 新建：Accelerate FSDP 配置
│   ├── grpo_gsm8k.yaml      # 新建
│   └── dpo_gsm8k.yaml       # 新建
├── tests/
│   ├── test_math_reward.py  # 新建
│   ├── test_utils.py        # 新建：sequence_logprobs
│   ├── test_dpo_loss.py     # 新建
│   └── test_grpo_loss.py    # 新建
├── scripts/
│   └── generate_gsm8k_pairs.py  # 新建：为 DPO 生成 preference pairs
├── train_grpo.py            # 新建：入口
├── train_dpo.py             # 新建：入口
└── requirements.txt         # 新建
```

---

## Task 1：项目配置

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `algorithms/__init__.py`, `data/__init__.py`, `rewards/__init__.py`, `trainer/__init__.py`, `eval/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.30.0
datasets>=2.18.0
wandb>=0.16.0
peft>=0.10.0
pytest>=8.0.0
```

- [ ] **Step 2: 创建 pytest.ini**

```ini
[pytest]
testpaths = tests
addopts = -v --tb=short
```

- [ ] **Step 3: 创建所有 __init__.py（空文件）**

```bash
mkdir -p algorithms data rewards trainer eval tests scripts configs
touch algorithms/__init__.py data/__init__.py rewards/__init__.py \
      trainer/__init__.py eval/__init__.py tests/__init__.py
```

- [ ] **Step 4: 安装依赖**

```bash
pip install -r requirements.txt
```

Expected: 无报错

- [ ] **Step 5: 验证导入**

```bash
python -c "import torch; import transformers; import accelerate; import datasets; import wandb; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini algorithms/ data/ rewards/ trainer/ eval/ tests/ scripts/ configs/
git commit -m "chore: project scaffold and dependencies"
```

---

## Task 2：Math Reward 函数

**Files:**
- Create: `rewards/math_reward.py`
- Create: `tests/test_math_reward.py`

- [ ] **Step 1: 写失败测试**

`tests/test_math_reward.py`:
```python
from rewards.math_reward import math_reward


def test_correct_answer():
    assert math_reward("Let me solve. 2x=4 #### 2", "2") == 1.0


def test_wrong_answer():
    assert math_reward("#### 5", "2") == 0.0


def test_no_marker():
    assert math_reward("The answer is two", "2") == 0.0


def test_decimal():
    assert math_reward("#### 3.14", "3.14") == 1.0


def test_whitespace_tolerance():
    assert math_reward("####  42 ", "42") == 1.0


def test_negative():
    assert math_reward("#### -7", "-7") == 1.0
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_math_reward.py -v
```

Expected: `ModuleNotFoundError: No module named 'rewards.math_reward'`

- [ ] **Step 3: 实现 math_reward**

`rewards/math_reward.py`:
```python
import re


def math_reward(response: str, ground_truth: str) -> float:
    """Extract answer after #### and compare with ground truth."""
    match = re.search(r"####\s*([\d\.\-]+)", response)
    if not match:
        return 0.0
    return 1.0 if match.group(1).strip() == ground_truth.strip() else 0.0
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_math_reward.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add rewards/math_reward.py tests/test_math_reward.py
git commit -m "feat: math reward with exact match on #### format"
```

---

## Task 3：GSM8K 数据加载器

**Files:**
- Create: `data/gsm8k.py`
- Create: `tests/test_gsm8k.py`

- [ ] **Step 1: 写失败测试**

`tests/test_gsm8k.py`:
```python
from data.gsm8k import load_gsm8k, extract_answer


def test_extract_answer():
    text = "Janet sells 16 eggs #### 16"
    assert extract_answer(text) == "16"


def test_extract_answer_missing():
    assert extract_answer("no answer here") is None


def test_load_gsm8k_train():
    ds = load_gsm8k(split="train")
    assert len(ds) > 7000
    row = ds[0]
    assert "prompt" in row
    assert "answer" in row
    assert row["answer"] is not None


def test_load_gsm8k_test():
    ds = load_gsm8k(split="test")
    assert len(ds) > 1000
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_gsm8k.py -v
```

- [ ] **Step 3: 实现 gsm8k.py**

`data/gsm8k.py`:
```python
import re
from datasets import load_dataset

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "At the end, write your final answer after ####.\n\n"
    "Problem: {question}\nSolution:"
)


def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\d\.\-,]+)", text)
    if not match:
        return None
    return match.group(1).replace(",", "").strip()


def load_gsm8k(split: str = "train") -> list[dict]:
    """Returns list of {prompt, answer} dicts."""
    raw = load_dataset("openai/gsm8k", "main", split=split)
    result = []
    for row in raw:
        answer = extract_answer(row["answer"])
        if answer is None:
            continue
        result.append({
            "prompt": PROMPT_TEMPLATE.format(question=row["question"]),
            "answer": answer,
        })
    return result
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_gsm8k.py -v
```

Expected: 4 passed（首次运行会下载数据集，需要网络）

- [ ] **Step 5: Commit**

```bash
git add data/gsm8k.py tests/test_gsm8k.py
git commit -m "feat: GSM8K data loader with prompt template"
```

---

## Task 4：Sequence Log Prob 工具函数

**Files:**
- Create: `algorithms/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: 写失败测试**

`tests/test_utils.py`:
```python
import torch
from algorithms.utils import sequence_logprobs


def _make_tiny_model():
    """Tiny GPT-2 style model for fast tests."""
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
    return AutoModelForCausalLM.from_config(config).eval()


def test_output_shape():
    model = _make_tiny_model()
    B, L = 2, 10
    input_ids = torch.randint(0, 100, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    response_mask = torch.zeros(B, L, dtype=torch.float)
    response_mask[:, 5:] = 1.0  # last 5 tokens are response

    logps = sequence_logprobs(model, input_ids, attention_mask, response_mask, no_grad=True)
    assert logps.shape == (B,)


def test_logps_are_negative():
    model = _make_tiny_model()
    B, L = 1, 8
    input_ids = torch.randint(0, 100, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    response_mask = torch.ones(B, L, dtype=torch.float)

    logps = sequence_logprobs(model, input_ids, attention_mask, response_mask, no_grad=True)
    assert (logps <= 0).all()


def test_empty_response_gives_zero():
    model = _make_tiny_model()
    B, L = 1, 8
    input_ids = torch.randint(0, 100, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    response_mask = torch.zeros(B, L, dtype=torch.float)  # all prompt, no response

    logps = sequence_logprobs(model, input_ids, attention_mask, response_mask, no_grad=True)
    assert logps[0].item() == 0.0
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_utils.py -v
```

- [ ] **Step 3: 实现 sequence_logprobs**

`algorithms/utils.py`:
```python
import contextlib
import torch
import torch.nn.functional as F
from torch import nn


def sequence_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,       # [B, seq_len]
    attention_mask: torch.Tensor,  # [B, seq_len]
    response_mask: torch.Tensor,   # [B, seq_len], 1.0 for response tokens
    no_grad: bool = False,
) -> torch.Tensor:                 # [B] sum of log probs over response tokens
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]              # [B, L-1, vocab]
    targets = input_ids[:, 1:]                      # [B, L-1]
    mask = response_mask[:, 1:].float()             # [B, L-1], shift to align with targets
    log_probs = F.log_softmax(logits, dim=-1)       # [B, L-1, vocab]
    token_logps = log_probs.gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)                                   # [B, L-1]
    return (token_logps * mask).sum(dim=-1)         # [B]
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_utils.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils.py tests/test_utils.py
git commit -m "feat: sequence_logprobs utility for DPO and GRPO"
```

---

## Task 5：DPO Loss

**Files:**
- Create: `algorithms/dpo.py`
- Create: `tests/test_dpo_loss.py`

- [ ] **Step 1: 写失败测试**

`tests/test_dpo_loss.py`:
```python
import torch
from algorithms.dpo import dpo_loss


def test_equal_policy_ref_gives_log2():
    # logits = 0 → loss = -log(sigmoid(0)) = log(2) ≈ 0.693
    t = torch.tensor([-1.0])
    loss, cr, rr = dpo_loss(t, t, t, t, beta=0.1)
    assert abs(loss.item() - 0.693) < 0.01


def test_preferred_chosen_gives_lower_loss():
    # policy strongly prefers chosen over rejected relative to ref
    policy_chosen   = torch.tensor([-0.5])
    policy_rejected = torch.tensor([-3.0])
    ref_chosen      = torch.tensor([-1.5])
    ref_rejected    = torch.tensor([-1.5])
    loss_good, _, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    # reversed: policy prefers rejected
    loss_bad, _, _ = dpo_loss(policy_rejected, policy_chosen, ref_chosen, ref_rejected)
    assert loss_good < loss_bad


def test_chosen_reward_higher_than_rejected():
    policy_chosen   = torch.tensor([-0.5])
    policy_rejected = torch.tensor([-2.0])
    ref_chosen      = torch.tensor([-1.0])
    ref_rejected    = torch.tensor([-1.0])
    _, chosen_r, rejected_r = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    assert chosen_r.item() > rejected_r.item()


def test_batch_shape():
    B = 4
    policy_chosen   = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen      = torch.randn(B)
    ref_rejected    = torch.randn(B)
    loss, cr, rr = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    assert loss.shape == ()   # scalar
    assert cr.shape == (B,)
    assert rr.shape == (B,)
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_dpo_loss.py -v
```

- [ ] **Step 3: 实现 dpo_loss**

`algorithms/dpo.py`:
```python
import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # [B]
    policy_rejected_logps: torch.Tensor,  # [B]
    ref_chosen_logps: torch.Tensor,       # [B]
    ref_rejected_logps: torch.Tensor,     # [B]
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (loss, chosen_rewards, rejected_rewards)
    chosen_rewards and rejected_rewards are detached scalars per sample.
    """
    # Log ratios relative to reference policy
    logr_chosen   = policy_chosen_logps   - ref_chosen_logps    # [B]
    logr_rejected = policy_rejected_logps - ref_rejected_logps  # [B]

    # DPO objective: maximize margin between chosen and rejected log ratios
    logits = beta * (logr_chosen - logr_rejected)               # [B]
    loss = -F.logsigmoid(logits).mean()

    # For logging: implicit reward estimates (detached)
    chosen_rewards   = (beta * logr_chosen).detach()
    rejected_rewards = (beta * logr_rejected).detach()

    return loss, chosen_rewards, rejected_rewards
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_dpo_loss.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add algorithms/dpo.py tests/test_dpo_loss.py
git commit -m "feat: DPO loss with log ratio contrastive objective"
```

---

## Task 6：GRPO Loss

**Files:**
- Create: `algorithms/grpo.py`
- Create: `tests/test_grpo_loss.py`

- [ ] **Step 1: 写失败测试**

`tests/test_grpo_loss.py`:
```python
import torch
from algorithms.grpo import grpo_loss, compute_advantages


def test_advantages_zero_when_all_same_reward():
    rewards = torch.ones(2, 4)
    advantages = compute_advantages(rewards)
    assert (advantages.abs() < 1e-6).all()


def test_advantages_normalized():
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    advantages = compute_advantages(rewards)
    assert abs(advantages.mean().item()) < 1e-5
    assert abs(advantages.std().item() - 1.0) < 0.01


def test_positive_advantage_for_higher_reward():
    rewards = torch.tensor([[1.0, 0.0]])
    advantages = compute_advantages(rewards)
    assert advantages[0, 0] > 0   # reward=1 → positive advantage
    assert advantages[0, 1] < 0   # reward=0 → negative advantage


def test_grpo_loss_shape():
    B, G = 2, 4
    log_probs = torch.randn(B, G)
    ref_log_probs = torch.randn(B, G)
    rewards = torch.rand(B, G)
    loss, advantages = grpo_loss(log_probs, ref_log_probs, rewards)
    assert loss.shape == ()          # scalar
    assert advantages.shape == (B, G)


def test_kl_beta_zero_ignores_ref():
    B, G = 1, 4
    log_probs = torch.tensor([[-0.5, -2.0, -0.3, -1.8]])
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    ref_same = log_probs.clone()
    ref_diff = torch.full((B, G), -5.0)

    loss_same, _ = grpo_loss(log_probs, ref_same, rewards, beta=0.0)
    loss_diff, _ = grpo_loss(log_probs, ref_diff, rewards, beta=0.0)
    assert abs(loss_same.item() - loss_diff.item()) < 1e-6
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_grpo_loss.py -v
```

- [ ] **Step 3: 实现 grpo.py**

`algorithms/grpo.py`:
```python
import torch


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Group-relative advantage normalization.
    rewards: [B, G]
    returns: [B, G], normalized advantages (mean=0, std=1 per group)
    """
    mean_r = rewards.mean(dim=-1, keepdim=True)             # [B, 1]
    std_r  = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return (rewards - mean_r) / std_r                       # [B, G]


def grpo_loss(
    log_probs: torch.Tensor,       # [B, G] sequence-level sum of log probs
    ref_log_probs: torch.Tensor,   # [B, G]
    rewards: torch.Tensor,         # [B, G]
    beta: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (total_loss, advantages)
    total_loss = policy_gradient_loss + beta * kl_penalty
    """
    advantages = compute_advantages(rewards)                 # [B, G]

    # Policy gradient: maximize advantage-weighted log prob
    pg_loss = -(advantages * log_probs).mean()

    # KL penalty: stay close to reference policy
    kl = (log_probs - ref_log_probs).mean()

    total_loss = pg_loss + beta * kl
    return total_loss, advantages
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_grpo_loss.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add algorithms/grpo.py tests/test_grpo_loss.py
git commit -m "feat: GRPO loss with group-relative advantage normalization"
```

---

## Task 7：Base Trainer（模型加载 + FSDP + W&B）

**Files:**
- Create: `trainer/base_trainer.py`

- [ ] **Step 1: 创建 base_trainer.py**

`trainer/base_trainer.py`:
```python
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrainerConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "checkpoints"
    run_name: str = "run"
    project: str = "llm-rl"
    lr: float = 1e-6
    batch_size: int = 4
    max_length: int = 1024
    max_new_tokens: int = 512
    eval_steps: int = 50
    save_steps: int = 200
    beta: float = 0.1
    seed: int = 42


class BaseTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.accelerator = Accelerator()
        self._setup_model()
        self._setup_ref_model()
        self._setup_wandb()

    def _setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr
        )
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

    def _setup_ref_model(self):
        """Load frozen reference model (full replica on each GPU, no FSDP)."""
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model = self.ref_model.to(self.accelerator.device)

    def _setup_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                project=self.config.project,
                name=self.config.run_name,
                config=vars(self.config),
            )

    def log(self, metrics: dict, step: int):
        if self.accelerator.is_main_process:
            wandb.log(metrics, step=step)

    def save_checkpoint(self, step: int):
        if self.accelerator.is_main_process:
            path = Path(self.config.output_dir) / f"step-{step}"
            path.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"Saved checkpoint to {path}")
```

- [ ] **Step 2: 验证导入（不启动训练）**

```bash
python -c "from trainer.base_trainer import BaseTrainer, TrainerConfig; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add trainer/base_trainer.py
git commit -m "feat: base trainer with FSDP, ref model, and W&B"
```

---

## Task 8：DPO Trainer

**Files:**
- Create: `trainer/dpo_trainer.py`

- [ ] **Step 1: 创建 dpo_trainer.py**

`trainer/dpo_trainer.py`:
```python
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from algorithms.dpo import dpo_loss
from algorithms.utils import sequence_logprobs
from trainer.base_trainer import BaseTrainer, TrainerConfig


@dataclass
class DPOConfig(TrainerConfig):
    beta: float = 0.1


class PreferenceDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data  # [{prompt, chosen, rejected}]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DPOTrainer(BaseTrainer):
    def __init__(self, config: DPOConfig, dataset: list[dict]):
        super().__init__(config)
        self.config = config
        ds = PreferenceDataset(dataset)
        self.dataloader = self.accelerator.prepare(
            DataLoader(ds, batch_size=config.batch_size, shuffle=True)
        )

    def _tokenize_pair(self, prompts, responses):
        """Tokenize prompt+response, return input_ids, attention_mask, response_mask."""
        pairs = [p + r for p, r in zip(prompts, responses)]
        enc = self.tokenizer(
            pairs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.accelerator.device)

        prompt_enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.accelerator.device)

        # response_mask: 1 where token belongs to response
        response_mask = torch.zeros_like(enc["attention_mask"], dtype=torch.float)
        for i, prompt_len in enumerate(prompt_enc["attention_mask"].sum(dim=-1)):
            response_mask[i, prompt_len:] = 1.0

        return enc["input_ids"], enc["attention_mask"], response_mask

    def train(self, num_steps: int):
        self.model.train()
        data_iter: Iterator = iter(self.dataloader)
        global_step = 0

        while global_step < num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            prompts   = batch["prompt"]
            chosens   = batch["chosen"]
            rejecteds = batch["rejected"]

            # Tokenize chosen and rejected
            chosen_ids, chosen_mask, chosen_resp_mask = self._tokenize_pair(prompts, chosens)
            rejected_ids, rejected_mask, rejected_resp_mask = self._tokenize_pair(prompts, rejecteds)

            # Compute log probs for training model
            policy_chosen_logps = sequence_logprobs(
                self.model, chosen_ids, chosen_mask, chosen_resp_mask
            )
            policy_rejected_logps = sequence_logprobs(
                self.model, rejected_ids, rejected_mask, rejected_resp_mask
            )

            # Compute log probs for ref model (no grad)
            ref_chosen_logps = sequence_logprobs(
                self.ref_model, chosen_ids, chosen_mask, chosen_resp_mask, no_grad=True
            )
            ref_rejected_logps = sequence_logprobs(
                self.ref_model, rejected_ids, rejected_mask, rejected_resp_mask, no_grad=True
            )

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=self.config.beta,
            )

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()

            self.log({
                "loss/train": loss.item(),
                "logr/chosen": chosen_rewards.mean().item(),
                "logr/rejected": rejected_rewards.mean().item(),
                "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            }, step=global_step)

            if global_step % self.config.save_steps == 0 and global_step > 0:
                self.save_checkpoint(global_step)

            global_step += 1
            if global_step % 10 == 0:
                print(f"step {global_step}/{num_steps}  loss={loss.item():.4f}")
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from trainer.dpo_trainer import DPOTrainer, DPOConfig; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add trainer/dpo_trainer.py
git commit -m "feat: DPO trainer with preference dataset and training loop"
```

---

## Task 9：生成 GSM8K Preference Pairs（DPO 数据）

**Files:**
- Create: `scripts/generate_gsm8k_pairs.py`

- [ ] **Step 1: 创建生成脚本**

`scripts/generate_gsm8k_pairs.py`:
```python
"""
用 SFT 模型对 GSM8K 每个 prompt 采样 2 条 response，
用 math_reward 打标，构建 (prompt, chosen, rejected) pairs。
保存为 data/gsm8k_preference_train.json。

Usage:
  python scripts/generate_gsm8k_pairs.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output data/gsm8k_preference_train.json \
    --num_samples 2000
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.gsm8k import load_gsm8k
from rewards.math_reward import math_reward


def generate_pair(model, tokenizer, prompt: str, answer: str, device: str) -> dict | None:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=2,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    responses = [
        tokenizer.decode(ids[inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        for ids in output_ids
    ]
    rewards = [math_reward(r, answer) for r in responses]
    if rewards[0] == rewards[1]:
        return None  # same reward → discard
    if rewards[0] > rewards[1]:
        chosen, rejected = responses[0], responses[1]
    else:
        chosen, rejected = responses[1], responses[0]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="data/gsm8k_preference_train.json")
    parser.add_argument("--num_samples", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    dataset = load_gsm8k(split="train")[:args.num_samples]
    pairs = []
    for i, row in enumerate(dataset):
        pair = generate_pair(model, tokenizer, row["prompt"], row["answer"], device)
        if pair:
            pairs.append(pair)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(dataset)}  collected={len(pairs)}")

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证导入**

```bash
python -c "import scripts.generate_gsm8k_pairs" 2>/dev/null || python scripts/generate_gsm8k_pairs.py --help
```

Expected: 显示 usage

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_gsm8k_pairs.py
git commit -m "feat: script to generate GSM8K preference pairs for DPO"
```

---

## Task 10：GRPO Trainer

**Files:**
- Create: `trainer/grpo_trainer.py`

- [ ] **Step 1: 创建 grpo_trainer.py**

`trainer/grpo_trainer.py`:
```python
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from algorithms.grpo import grpo_loss, compute_advantages
from algorithms.utils import sequence_logprobs
from rewards.math_reward import math_reward
from trainer.base_trainer import BaseTrainer, TrainerConfig


@dataclass
class GRPOConfig(TrainerConfig):
    beta: float = 0.04
    group_size: int = 8           # G: responses per prompt
    temperature: float = 0.8
    reward_fn_name: str = "math"  # "math" only in Phase 1


class PromptDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data  # [{prompt, answer}]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GRPOTrainer(BaseTrainer):
    def __init__(self, config: GRPOConfig, dataset: list[dict]):
        super().__init__(config)
        self.config = config
        ds = PromptDataset(dataset)
        self.dataloader = self.accelerator.prepare(
            DataLoader(ds, batch_size=config.batch_size, shuffle=True)
        )

    def _rollout(self, prompts: list[str]) -> list[list[str]]:
        """Generate G responses per prompt using model.generate()."""
        G = self.config.group_size
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.accelerator.device)

        unwrapped = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            output_ids = unwrapped.generate(
                **enc,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=G,
                do_sample=True,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # output_ids: [B*G, seq_len]
        prompt_len = enc["input_ids"].shape[-1]
        responses_flat = [
            self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            for ids in output_ids
        ]
        # Reshape to [B, G]
        B = len(prompts)
        return [responses_flat[i * G:(i + 1) * G] for i in range(B)]

    def _compute_rewards(
        self, prompts: list[str], responses: list[list[str]], answers: list[str]
    ) -> torch.Tensor:
        """Compute scalar reward per (prompt, response). Returns [B, G]."""
        B, G = len(prompts), self.config.group_size
        rewards = torch.zeros(B, G, device=self.accelerator.device)
        for i, (resps, ans) in enumerate(zip(responses, answers)):
            for j, resp in enumerate(resps):
                rewards[i, j] = math_reward(resp, ans)
        return rewards

    def _compute_logprobs(
        self, prompts: list[str], responses: list[list[str]], no_grad: bool = False
    ) -> torch.Tensor:
        """Compute sequence log probs. Returns [B, G]."""
        G = self.config.group_size
        # Flatten [B*G] pairs for batched forward pass
        flat_pairs = [
            p + r
            for p, resps in zip(prompts, responses)
            for r in resps
        ]
        flat_prompts = [p for p in prompts for _ in range(G)]
        enc = self.tokenizer(
            flat_pairs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.accelerator.device)
        prompt_enc = self.tokenizer(
            flat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.accelerator.device)

        # Build response mask
        response_mask = torch.zeros_like(enc["attention_mask"], dtype=torch.float)
        for i, plen in enumerate(prompt_enc["attention_mask"].sum(dim=-1)):
            response_mask[i, plen:] = 1.0

        model = self.ref_model if no_grad else self.model
        flat_logps = sequence_logprobs(
            model, enc["input_ids"], enc["attention_mask"], response_mask, no_grad=no_grad
        )  # [B*G]
        return flat_logps.view(len(prompts), G)   # [B, G]

    def train(self, num_steps: int):
        self.model.train()
        data_iter: Iterator = iter(self.dataloader)
        global_step = 0

        while global_step < num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            prompts = batch["prompt"]
            answers = batch["answer"]

            # ① Rollout: generate G responses per prompt
            responses = self._rollout(list(prompts))

            # ② Score with reward function
            rewards = self._compute_rewards(list(prompts), responses, list(answers))  # [B, G]

            # ③ Compute log probs (training model + ref model)
            log_probs = self._compute_logprobs(list(prompts), responses, no_grad=False)  # [B, G]
            ref_log_probs = self._compute_logprobs(list(prompts), responses, no_grad=True)

            # ④ GRPO loss
            loss, advantages = grpo_loss(log_probs, ref_log_probs, rewards, beta=self.config.beta)

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()

            self.log({
                "loss/train": loss.item(),
                "reward/mean": rewards.mean().item(),
                "reward/std": rewards.std().item(),
                "reward/pass_rate": (rewards == 1.0).float().mean().item(),
                "advantage/mean": advantages.mean().item(),
                "kl/mean": (log_probs - ref_log_probs).mean().item(),
            }, step=global_step)

            if global_step % self.config.save_steps == 0 and global_step > 0:
                self.save_checkpoint(global_step)

            global_step += 1
            if global_step % 10 == 0:
                print(
                    f"step {global_step}/{num_steps}  "
                    f"loss={loss.item():.4f}  "
                    f"reward={rewards.mean().item():.3f}"
                )
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from trainer.grpo_trainer import GRPOTrainer, GRPOConfig; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add trainer/grpo_trainer.py
git commit -m "feat: GRPO trainer with model.generate() rollout and group advantage"
```

---

## Task 11：Configs + Entry Points

**Files:**
- Create: `configs/fsdp_4gpu.yaml`
- Create: `configs/grpo_gsm8k.yaml`
- Create: `configs/dpo_gsm8k.yaml`
- Create: `train_grpo.py`
- Create: `train_dpo.py`

- [ ] **Step 1: 创建 fsdp_4gpu.yaml**

`configs/fsdp_4gpu.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
```

- [ ] **Step 2: 创建 grpo_gsm8k.yaml**

`configs/grpo_gsm8k.yaml`:
```yaml
model_name: Qwen/Qwen2.5-7B-Instruct
run_name: grpo-gsm8k
project: llm-rl
output_dir: checkpoints/grpo-gsm8k
lr: 1.0e-6
batch_size: 4
group_size: 8
beta: 0.04
max_length: 1024
max_new_tokens: 512
temperature: 0.8
eval_steps: 50
save_steps: 200
num_steps: 1000
```

- [ ] **Step 3: 创建 dpo_gsm8k.yaml**

`configs/dpo_gsm8k.yaml`:
```yaml
model_name: Qwen/Qwen2.5-7B-Instruct
run_name: dpo-gsm8k
project: llm-rl
output_dir: checkpoints/dpo-gsm8k
lr: 5.0e-7
batch_size: 16
beta: 0.1
max_length: 1024
eval_steps: 50
save_steps: 200
num_steps: 1000
data_path: data/gsm8k_preference_train.json
```

- [ ] **Step 4: 创建 train_grpo.py**

`train_grpo.py`:
```python
import argparse
import yaml

from data.gsm8k import load_gsm8k
from trainer.grpo_trainer import GRPOConfig, GRPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_gsm8k.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    config = GRPOConfig(**{k: v for k, v in cfg.items() if k != "num_steps"})
    num_steps = cfg.get("num_steps", 1000)

    dataset = load_gsm8k(split="train")
    trainer = GRPOTrainer(config, dataset)
    trainer.train(num_steps=num_steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 创建 train_dpo.py**

`train_dpo.py`:
```python
import argparse
import json
import yaml

from trainer.dpo_trainer import DPOConfig, DPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_gsm8k.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = cfg.pop("data_path")
    config = DPOConfig(**{k: v for k, v in cfg.items() if k != "num_steps"})
    num_steps = cfg.get("num_steps", 1000)

    with open(data_path) as f:
        dataset = json.load(f)

    trainer = DPOTrainer(config, dataset)
    trainer.train(num_steps=num_steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 验证 CLI 可以启动**

```bash
python train_grpo.py --help
python train_dpo.py --help
```

Expected: 显示 usage，无 ImportError

- [ ] **Step 7: Commit**

```bash
git add configs/ train_grpo.py train_dpo.py
git commit -m "feat: configs and training entry points"
```

---

## Task 12：GSM8K 评估

**Files:**
- Create: `eval/gsm8k_eval.py`

- [ ] **Step 1: 创建 gsm8k_eval.py**

`eval/gsm8k_eval.py`:
```python
"""
Evaluate a model on GSM8K test set.

Usage:
  python eval/gsm8k_eval.py --model checkpoints/grpo-gsm8k/step-200
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.gsm8k import load_gsm8k
from rewards.math_reward import math_reward


def evaluate(model_path: str, num_examples: int = 500) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()

    dataset = load_gsm8k(split="test")[:num_examples]
    correct = 0

    for i, row in enumerate(dataset):
        enc = tokenizer(row["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,        # greedy for eval
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            output_ids[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        if math_reward(response, row["answer"]) == 1.0:
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(dataset)}  acc={correct/(i+1):.3f}")

    accuracy = correct / len(dataset)
    print(f"\nFinal accuracy: {accuracy:.4f} ({correct}/{len(dataset)})")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_examples", type=int, default=500)
    args = parser.parse_args()
    evaluate(args.model, args.num_examples)
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from eval.gsm8k_eval import evaluate; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add eval/gsm8k_eval.py
git commit -m "feat: GSM8K test set evaluation with greedy decoding"
```

---

## Task 13：端到端 Smoke Test

**Files:**
- Create: `scripts/smoke_test.py`

- [ ] **Step 1: 创建 smoke test**

`scripts/smoke_test.py`:
```python
"""
3-step smoke test on tiny model. No GPU needed.
Verifies the full training pipeline without Qwen.

Usage: python scripts/smoke_test.py
"""

import json
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Use tiny-gpt2 so the test runs in <60s on CPU
TINY_MODEL = "sshleifer/tiny-gpt2"


def test_dpo_smoke():
    print("\n=== DPO Smoke Test ===")
    from trainer.dpo_trainer import DPOConfig, DPOTrainer

    dataset = [
        {"prompt": "Q: 1+1=?", "chosen": " 2 #### 2", "rejected": " 3 #### 3"},
        {"prompt": "Q: 2+2=?", "chosen": " 4 #### 4", "rejected": " 5 #### 5"},
        {"prompt": "Q: 3+3=?", "chosen": " 6 #### 6", "rejected": " 7 #### 7"},
        {"prompt": "Q: 4+4=?", "chosen": " 8 #### 8", "rejected": " 9 #### 9"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = DPOConfig(
            model_name=TINY_MODEL,
            output_dir=tmpdir,
            run_name="smoke-dpo",
            lr=1e-4,
            batch_size=2,
            eval_steps=999,
            save_steps=999,
        )
        trainer = DPOTrainer(config, dataset)
        trainer.train(num_steps=3)
    print("DPO smoke test PASSED")


def test_grpo_smoke():
    print("\n=== GRPO Smoke Test ===")
    from trainer.grpo_trainer import GRPOConfig, GRPOTrainer

    dataset = [
        {"prompt": "Q: 1+1=? #### ", "answer": "2"},
        {"prompt": "Q: 2+2=? #### ", "answer": "4"},
        {"prompt": "Q: 3+3=? #### ", "answer": "6"},
        {"prompt": "Q: 4+4=? #### ", "answer": "8"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GRPOConfig(
            model_name=TINY_MODEL,
            output_dir=tmpdir,
            run_name="smoke-grpo",
            lr=1e-4,
            batch_size=2,
            group_size=2,
            max_new_tokens=16,
            eval_steps=999,
            save_steps=999,
        )
        trainer = GRPOTrainer(config, dataset)
        trainer.train(num_steps=3)
    print("GRPO smoke test PASSED")


if __name__ == "__main__":
    # Disable W&B for smoke test
    import os
    os.environ["WANDB_MODE"] = "disabled"
    test_dpo_smoke()
    test_grpo_smoke()
    print("\nAll smoke tests PASSED")
```

- [ ] **Step 2: 运行 smoke test（CPU，tiny model）**

```bash
WANDB_MODE=disabled python scripts/smoke_test.py
```

Expected:
```
=== DPO Smoke Test ===
step 3/3  loss=...
DPO smoke test PASSED

=== GRPO Smoke Test ===
step 3/3  loss=...  reward=...
GRPO smoke test PASSED

All smoke tests PASSED
```

- [ ] **Step 3: 运行全部单元测试**

```bash
pytest tests/ -v
```

Expected: 所有测试通过

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "test: end-to-end smoke test for DPO and GRPO trainers"
```

---

## Task 14：正式训练（Phase 1）

**前提条件：**
1. Task 9 的 preference pairs 已生成（`data/gsm8k_preference_train.json`）
2. W&B 已登录（`wandb login`）

- [ ] **Step 1: 生成 DPO 训练数据**

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/generate_gsm8k_pairs.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output data/gsm8k_preference_train.json \
  --num_samples 3000
```

Expected: `Saved ~1500 pairs to data/gsm8k_preference_train.json`（约一半样本会因两次生成结果相同而丢弃）

- [ ] **Step 2: 启动 DPO 训练（Run B）**

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --config_file configs/fsdp_4gpu.yaml \
  train_dpo.py \
  --config configs/dpo_gsm8k.yaml
```

- [ ] **Step 3: 启动 GRPO 训练（Run A，新终端）**

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --config_file configs/fsdp_4gpu.yaml \
  train_grpo.py \
  --config configs/grpo_gsm8k.yaml
```

- [ ] **Step 4: 训练结束后评估**

```bash
# 评估 GRPO checkpoint
CUDA_VISIBLE_DEVICES=4 python eval/gsm8k_eval.py \
  --model checkpoints/grpo-gsm8k/step-1000

# 评估 DPO checkpoint
CUDA_VISIBLE_DEVICES=4 python eval/gsm8k_eval.py \
  --model checkpoints/dpo-gsm8k/step-1000
```

- [ ] **Step 5: 在 W&B 对比曲线**

打开 W&B project `llm-rl`，Compare Runs，观察：
- `reward/mean`（GRPO）vs `reward_margin`（DPO）
- `kl/mean` 两条曲线的散度
- `eval/accuracy` 最终指标差异

---

## 自检结果

| 检查项 | 状态 |
|---|---|
| Placeholder 扫描 | 无 TBD/TODO ✓ |
| 类型一致性 | `sequence_logprobs` 签名在 Task 4 定义，Task 8/10 完全一致 ✓ |
| Spec 覆盖：算法核心 | Task 5/6 ✓ |
| Spec 覆盖：数据加载 | Task 3/9 ✓ |
| Spec 覆盖：分布式训练 | Task 7/11（fsdp_4gpu.yaml）✓ |
| Spec 覆盖：实验追踪 | Task 7（W&B）、Task 8/10（log 指标）✓ |
| Spec 覆盖：评估 | Task 12 ✓ |
| Spec 覆盖：Smoke test | Task 13 ✓ |
| Phase 2 | 不在范围内，独立计划 |
