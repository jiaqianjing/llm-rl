"""
3-step smoke test on tiny model. No GPU needed.
Verifies the full training pipeline without Qwen.

Usage: python scripts/smoke_test.py
"""

import os
import sys
from pathlib import Path

# Add project root to path so modules can be imported when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable W&B for smoke test
os.environ["WANDB_MODE"] = "disabled"
# Force CPU so the tiny-gpt2 model (head_dim=1) works without CUBLAS issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_dpo_smoke():
    print("\n=== DPO Smoke Test ===")
    from trainer.dpo_trainer import DPOConfig, DPOTrainer

    dataset = [
        {"prompt": "Q: 1+1=?", "chosen": " 2 #### 2", "rejected": " 3 #### 3"},
        {"prompt": "Q: 2+2=?", "chosen": " 4 #### 4", "rejected": " 5 #### 5"},
        {"prompt": "Q: 3+3=?", "chosen": " 6 #### 6", "rejected": " 7 #### 7"},
        {"prompt": "Q: 4+4=?", "chosen": " 8 #### 8", "rejected": " 9 #### 9"},
    ]

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config = DPOConfig(
            model_name="sshleifer/tiny-gpt2",
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

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config = GRPOConfig(
            model_name="sshleifer/tiny-gpt2",
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
    test_dpo_smoke()
    test_grpo_smoke()
    print("\nAll smoke tests PASSED")
