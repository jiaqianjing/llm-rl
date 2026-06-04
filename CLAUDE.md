# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

From-scratch implementations of GRPO and DPO on Qwen2.5-7B-Instruct, targeting GSM8K math reasoning (Phase 1) and UltraFeedback instruction following (Phase 2). Hardware: 4x A100 80G (GPUs 4-7), Accelerate FSDP ZeRO-3.

## Commands

```bash
# Run unit tests (pure math, no GPU needed)
pytest

# Run a single test file
pytest tests/test_grpo_loss.py

# Run the full pipeline smoke test (CPU, tiny-gpt2, ~30s)
python scripts/smoke_test.py

# Train GRPO on GSM8K (4 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/mnt/nvme6/ken/opt-pytorch-pkgs:/opt/pytorch/lib/python3.12/site-packages \
/opt/pytorch/bin/python3 -m accelerate.commands.accelerate_cli launch \
  --config_file configs/fsdp_4gpu.yaml train_grpo.py --config configs/grpo_gsm8k.yaml

# Train DPO on GSM8K (4 GPUs) — requires data/gsm8k_preference_train.json
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/mnt/nvme6/ken/opt-pytorch-pkgs:/opt/pytorch/lib/python3.12/site-packages \
/opt/pytorch/bin/python3 -m accelerate.commands.accelerate_cli launch \
  --config_file configs/fsdp_4gpu.yaml train_dpo.py --config configs/dpo_gsm8k.yaml

# Generate DPO preference pairs from GSM8K (4-GPU parallel, ~1000 pairs from 7473 prompts)
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((shard+4)) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=/mnt/nvme6/ken/opt-pytorch-pkgs:/opt/pytorch/lib/python3.12/site-packages \
  /opt/pytorch/bin/python3 scripts/generate_gsm8k_pairs.py \
    --model Qwen/Qwen2.5-7B-Instruct --output data/gsm8k_preference_train.json \
    --num_samples 7473 --batch_size 8 --shard $shard --num_shards 4 &
done

# Evaluate a checkpoint on GSM8K test set (greedy decoding)
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/mnt/nvme6/ken/opt-pytorch-pkgs:/opt/pytorch/lib/python3.12/site-packages:/mnt/nvme6/ken/rl/llm-rl \
/opt/pytorch/bin/python3 eval/gsm8k_eval.py --model checkpoints/dpo-gsm8k/step-1000
```

## Architecture

### Core algorithm layer (`algorithms/`)
Pure PyTorch, no model loading — these are the math kernels:
- `grpo.py`: `compute_advantages([B,G] rewards → normalized)` + `grpo_loss(log_probs, ref_log_probs, rewards, beta)` → PG loss + KL penalty
- `dpo.py`: `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)` → BCE on log-ratio margin
- `utils.py`: `sequence_logprobs(model, input_ids, attention_mask, response_mask)` — shared forward pass that sums log probs over response tokens only (shifts by 1 to align logits→targets)

### Trainer layer (`trainer/`)
- `base_trainer.py`: loads policy model + frozen ref model, wraps policy with Accelerate, initializes W&B on main process only. `TrainerConfig` fields control dtype, attn_implementation, and all hyperparams.
- `grpo_trainer.py`: `GRPOTrainer` — rollout → reward → logprob → GRPO loss. Key: `_rollout` uses `model.generate()` with `num_return_sequences=G` to sample G completions per prompt, then batches them for logprob computation.
- `dpo_trainer.py`: `DPOTrainer` — tokenizes chosen/rejected pairs, computes policy and ref log probs, applies DPO loss.

**Shared pattern**: both trainers call `sequence_logprobs` for the ref model with `no_grad=True`. Response masking is applied by computing prompt length from a separate tokenization of the prompt-only text, then setting `response_mask[i, prompt_len:]= 1.0`.

### Data layer (`data/`, `rewards/`)
- `data/gsm8k.py`: loads HuggingFace `openai/gsm8k`, returns `{question, answer}` dicts; answer is numeric string extracted after `####`
- `rewards/math_reward.py`: binary reward — tries `####` format first, fallback to last number in response
- `scripts/generate_gsm8k_pairs.py`: samples 4 responses per prompt (2×2 batches to avoid KV-cache OOM), uses `math_reward` to pick best/worst, outputs `{prompt, chosen, rejected}` JSON. Prompt uses chat template + `####` format instruction. Supports `--shard`/`--num_shards` for multi-GPU parallel generation.

### Config / entry points
- YAML configs in `configs/` map directly to dataclass fields (`GRPOConfig`, `DPOConfig`). `num_steps` and `data_path` are popped before constructing the dataclass.
- `configs/fsdp_4gpu.yaml`: Accelerate FSDP config for 4-GPU full sharding with bf16 mixed precision.

## Runtime Environment

**Working Python/PyTorch:** `/opt/pytorch/bin/python3` (PyTorch 2.8.0+cu129, NCCL 2.27.3)
- System NCCL at `/usr/local/cuda/lib/libnccl.so.2.27.3` is the only working NCCL on this machine
- pip-installed NCCL (2.28+) causes SIGSEGV on all GPU collectives — see `docs/env-pitfalls.md`
- Extra packages installed at `/mnt/nvme6/ken/opt-pytorch-pkgs` (transformers, accelerate, etc.)
- Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce OOM

**GPUs:** Use GPUs 4-7 for training (`CUDA_VISIBLE_DEVICES=4,5,6,7`), GPU 1 or 3 for eval/data gen.

## Key Design Decisions

- The ref model is a full replica loaded with `.from_pretrained()` — it is NOT wrapped by FSDP, just moved to device. This is intentional: avoids FSDP complexity for a frozen model.
- `sequence_logprobs` does a left-shift: `logits[:, :-1, :]` vs `input_ids[:, 1:]` so each position predicts the next token. Response mask is shifted identically.
- GRPO `_compute_logprobs` flattens `[B, G]` pairs into `[B*G]` for a single batched forward, then reshapes back.
- `torch_dtype` defaults to `bfloat16` on CUDA, `float32` on CPU; override via config for testing with tiny models.
