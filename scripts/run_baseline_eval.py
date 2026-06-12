"""
Run the baseline (no-RL) GSM8K eval and log results + per-example logs to an
existing W&B run (overwrites it).

Reuses the SINGLE SOURCE OF TRUTH for answer extraction (rewards.math_reward)
and the test-set loader + generation params from eval/gsm8k_eval.py so this
matches exactly what training will be scored against.

Usage (GPU 1, overwrite run hhgaqxpe):
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=/mnt/nvme6/ken/opt-pytorch-pkgs:/opt/pytorch/lib/python3.12/site-packages:/mnt/nvme6/ken/rl/llm-rl \
  /opt/pytorch/bin/python3 scripts/run_baseline_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --wandb_run_id hhgaqxpe --wandb_entity jiaqianjing --wandb_project llm-rl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

# Single source of truth — same functions the reward fn and the evaluator use.
from rewards.math_reward import extract_answer
from eval.gsm8k_eval import load_gsm8k_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--wandb_run_id", required=True)
    parser.add_argument("--wandb_project", default="llm-rl")
    parser.add_argument("--wandb_entity", default="jiaqianjing")
    parser.add_argument("--log_dir", default="eval_logs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    dataset = load_gsm8k_test(args.num_examples)
    print(f"loaded {len(dataset)} gsm8k-test examples", flush=True)

    records = []          # per-example logs
    correct = 0
    t0 = time.time()

    for i, row in enumerate(dataset):
        messages = [{"role": "user", "content": row["question"]
                     + "\n\nSolve step by step. Write your final answer after ####."}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        pred = extract_answer(response)
        is_correct = pred is not None and pred == row["answer"]
        correct += int(is_correct)

        records.append({
            "idx": i,
            "question": row["question"],
            "gold": row["answer"],
            "pred": pred,
            "correct": is_correct,
            "response": response,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"{i+1}/{len(dataset)}  acc={correct/(i+1):.3f}  "
                  f"({elapsed:.0f}s, {elapsed/(i+1):.1f}s/ex)", flush=True)

    accuracy = correct / len(dataset)
    elapsed = time.time() - t0
    print(f"\nFinal accuracy: {accuracy:.4f} ({correct}/{len(dataset)})  in {elapsed:.0f}s", flush=True)

    # --- write per-example logs locally (kept for inspection) ---
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    local_jsonl = log_dir / f"baseline_{args.wandb_run_id}_gsm8k.jsonl"
    with open(local_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote per-example log -> {local_jsonl}", flush=True)

    # --- log to W&B (overwrite the existing finished run) ---
    import wandb
    run = wandb.init(
        id=args.wandb_run_id, resume="must",
        project=args.wandb_project, entity=args.wandb_entity,
    )
    # Match the existing run's config schema, overwrite values.
    run.config.update({
        "model": args.model,
        "correct": correct,
        "decoding": "greedy",
        "eval_set": "gsm8k-test",
        "num_examples": args.num_examples,
        "prompt_format": "chat_template + #### instruction",
        "max_new_tokens": args.max_new_tokens,
    }, allow_val_change=True)

    run.summary["eval/gsm8k_accuracy"] = accuracy
    run.summary["eval/gsm8k_correct"] = correct
    run.summary["eval/gsm8k_total"] = len(dataset)

    # Attach the per-example log + a one-table artifact so it shows in the run UI.
    run_jsonl = Path(run.dir) / "eval_gsm8k_per_example.jsonl"
    run_jsonl.write_text(local_jsonl.read_text())
    wandb.save(str(run_jsonl), base_path=run.dir, policy="now")

    table = wandb.Table(columns=["idx", "gold", "pred", "correct", "response"])
    for r in records:
        table.add_data(r["idx"], r["gold"], str(r["pred"]), r["correct"], r["response"])
    run.log({"eval/gsm8k_examples": table})

    print(f"logged to {run.url}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
