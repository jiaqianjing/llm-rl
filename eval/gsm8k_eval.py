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
