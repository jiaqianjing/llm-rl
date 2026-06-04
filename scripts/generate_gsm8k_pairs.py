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
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path so modules can be imported when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.gsm8k import load_gsm8k
from rewards.math_reward import math_reward


def generate_pair(model, tokenizer, question: str, answer: str, device,
                  num_samples: int = 4) -> dict | None:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    # Split into batches of 2 to avoid KV-cache OOM with large num_return_sequences
    batch_size = 2
    responses = []
    for _ in range(num_samples // batch_size):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=batch_size,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        responses += [
            tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            for ids in output_ids
        ]

    rewards = [math_reward(r, answer) for r in responses]
    best_reward, worst_reward = max(rewards), min(rewards)
    if best_reward == worst_reward:
        return None  # all same reward → no signal

    chosen   = responses[rewards.index(best_reward)]
    rejected = responses[len(rewards) - 1 - rewards[::-1].index(worst_reward)]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="data/gsm8k_preference_train.json")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device_map
    )
    model.eval()
    device = next(model.parameters()).device

    dataset = load_gsm8k(split="train")[:args.num_samples]
    dataset = dataset[args.shard::args.num_shards]   # interleaved sharding
    print(f"shard {args.shard}/{args.num_shards}: {len(dataset)} samples", flush=True)

    pairs = []
    for i, row in enumerate(dataset):
        pair = generate_pair(model, tokenizer, row["question"], row["answer"], device, num_samples=4)
        if pair:
            pairs.append(pair)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(dataset)}  collected={len(pairs)}", flush=True)

    out_path = Path(args.output)
    if args.num_shards > 1:
        out_path = out_path.with_suffix(f".shard{args.shard}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pairs)} pairs to {out_path}", flush=True)


if __name__ == "__main__":
    main()
