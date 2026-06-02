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
