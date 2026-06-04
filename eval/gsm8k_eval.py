"""
Evaluate a model on GSM8K test set using chat template (official-style).

Usage:
  python eval/gsm8k_eval.py --model Qwen/Qwen2.5-7B-Instruct
  python eval/gsm8k_eval.py --model checkpoints/dpo-gsm8k/step-1000
"""

import argparse
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gsm8k_test(num_examples: int) -> list[dict]:
    raw = load_dataset("openai/gsm8k", "main", split="test")
    result = []
    for row in raw:
        # Extract ground truth from GSM8K's "#### <answer>" format
        m = re.search(r"####\s*([\d,\.]+)", row["answer"])
        if m is None:
            continue
        result.append({
            "question": row["question"],
            "answer": m.group(1).replace(",", "").strip(),
        })
    return result[:num_examples]


def extract_answer(text: str) -> str | None:
    """
    Two-pass extraction:
    1. GSM8K-style  #### <number>
    2. Fallback: last standalone number in the response
    """
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # fallback: last number that looks like a final answer
    numbers = re.findall(r"(?<![/\d])(-?\d{1,10}(?:\.\d+)?)(?!\d)", text)
    return numbers[-1].replace(",", "") if numbers else None


def evaluate(model_path: str, num_examples: int = 500) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    dataset = load_gsm8k_test(num_examples)
    correct = 0

    for i, row in enumerate(dataset):
        # Official-style: use chat template, no custom prompt wrapping
        messages = [{"role": "user", "content": row["question"] + "\n\nSolve step by step. Write your final answer after ####."}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        pred = extract_answer(response)
        if pred is not None and pred == row["answer"]:
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
