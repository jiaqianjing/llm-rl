"""
用 SFT 模型对 GSM8K 每个 prompt 采样 4 条 response（批量生成），
用 math_reward 打标，取最好/最差构建 (prompt, chosen, rejected) pairs。
保存为 data/gsm8k_preference_train.json。

Usage:
  python scripts/generate_gsm8k_pairs.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output data/gsm8k_preference_train.json \
    --num_samples 7473 \
    --batch_size 8
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.gsm8k import load_gsm8k
from rewards.math_reward import math_reward


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question + "\n\nSolve step by step. Write your final answer after ####."}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch(model, tokenizer, questions: list[str], answers: list[str],
                   device, num_samples: int = 4) -> list[dict | None]:
    """
    Generate num_samples responses for each question in the batch.
    Returns a list of (prompt, chosen, rejected) dicts or None per question.
    """
    prompts = [build_prompt(tokenizer, q) for q in questions]
    B = len(prompts)

    # Tokenize with padding
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=1024,
    ).to(device)
    prompt_lens = enc["attention_mask"].sum(dim=-1).tolist()

    input_len = enc["input_ids"].shape[-1]  # padded input length (same for all in batch)

    # Two generate calls of num_return_sequences=2 to avoid KV-cache OOM
    all_responses = [[] for _ in range(B)]
    for _ in range(num_samples // 2):
        with torch.no_grad():
            output_ids = model.generate(
                **enc,
                max_new_tokens=512,
                num_return_sequences=2,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        # output_ids: [B*2, input_len + generated_len]
        # generated tokens start at input_len (left-padding means all inputs same length)
        for b in range(B):
            for k in range(2):
                idx = b * 2 + k
                resp = tokenizer.decode(
                    output_ids[idx][input_len:], skip_special_tokens=True
                )
                all_responses[b].append(resp)

    results = []
    for b, (prompt, answer, responses) in enumerate(zip(prompts, answers, all_responses)):
        rewards = [math_reward(r, answer) for r in responses]
        best_r, worst_r = max(rewards), min(rewards)
        if best_r == worst_r:
            results.append(None)
            continue
        chosen   = responses[rewards.index(best_r)]
        rejected = responses[len(rewards) - 1 - rewards[::-1].index(worst_r)]
        results.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="data/gsm8k_preference_train.json")
    parser.add_argument("--num_samples", type=int, default=7473)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # decoder-only 模型生成时用 left padding
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device_map
    )
    model.eval()
    device = next(model.parameters()).device

    dataset = load_gsm8k(split="train")[:args.num_samples]
    dataset = dataset[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(dataset)} samples", flush=True)

    pairs = []
    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i: i + args.batch_size]
        questions = [row["question"] for row in batch]
        answers   = [row["answer"]   for row in batch]

        results = generate_batch(model, tokenizer, questions, answers, device, num_samples=4)
        pairs.extend([r for r in results if r is not None])

        processed = min(i + args.batch_size, len(dataset))
        if processed % 100 == 0 or processed == len(dataset):
            print(f"{processed}/{len(dataset)}  collected={len(pairs)}", flush=True)

    out_path = Path(args.output)
    if args.num_shards > 1:
        out_path = out_path.with_suffix(f".shard{args.shard}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pairs)} pairs to {out_path}", flush=True)


if __name__ == "__main__":
    main()
