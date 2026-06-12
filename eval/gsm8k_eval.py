"""
Evaluate a model on GSM8K test set using chat template (official-style).

Usage:
  python eval/gsm8k_eval.py --model Qwen/Qwen2.5-7B-Instruct
  python eval/gsm8k_eval.py --model checkpoints/dpo-gsm8k/step-1000
  python eval/gsm8k_eval.py --model checkpoints/dpo-gsm8k/step-1000 --wandb_run_id moyoagq9
"""

import argparse
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Single source of truth for answer extraction — shared with the reward function
# so DPO data generation, GRPO rollout scoring, and eval all agree.
from rewards.math_reward import extract_answer, normalize_num


def load_gsm8k_test(num_examples: int) -> list[dict]:
    raw = load_dataset("openai/gsm8k", "main", split="test")
    result = []
    for row in raw:
        # Extract ground truth from GSM8K's "#### <answer>" format
        m = re.search(r"####\s*([\d,\.\-]+)", row["answer"])
        if m is None:
            continue
        result.append({
            "question": row["question"],
            "answer": normalize_num(m.group(1)),
        })
    return result[:num_examples]


def evaluate(model_path: str, num_examples: int = 500, wandb_run_id: str | None = None,
             wandb_project: str = "llm-rl", wandb_entity: str | None = None) -> float:
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
                max_new_tokens=768,
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

    if wandb_run_id is not None:
        import wandb
        wandb.init(id=wandb_run_id, resume="must", project=wandb_project, entity=wandb_entity)
        wandb.log({"eval/gsm8k_accuracy": accuracy, "eval/gsm8k_correct": correct, "eval/gsm8k_total": len(dataset)})
        wandb.finish()

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--wandb_run_id", default=None, help="Resume an existing W&B run and log eval metrics to it")
    parser.add_argument("--wandb_project", default="llm-rl")
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args()
    evaluate(args.model, args.num_examples, args.wandb_run_id, args.wandb_project, args.wandb_entity)
