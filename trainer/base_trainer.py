import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    # None means auto-select: bfloat16 if CUDA available, else float32
    torch_dtype: Optional[str] = None
    # Attention implementation: "sdpa" (default), "eager", "flash_attention_2"
    attn_implementation: Optional[str] = None


class BaseTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.accelerator = Accelerator()
        self._setup_model()
        self._setup_ref_model()
        self._setup_wandb()

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve model dtype: explicit config > auto (bfloat16 on CUDA, float32 on CPU)."""
        if self.config.torch_dtype is not None:
            return getattr(torch, self.config.torch_dtype)
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = self._resolve_dtype()
        load_kwargs = dict(dtype=dtype)
        if self.config.attn_implementation is not None:
            load_kwargs["attn_implementation"] = self.config.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr
        )
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

    def _setup_ref_model(self):
        """Load frozen reference model (full replica on each GPU, no FSDP)."""
        dtype = self._resolve_dtype()
        load_kwargs = dict(dtype=dtype)
        if self.config.attn_implementation is not None:
            load_kwargs["attn_implementation"] = self.config.attn_implementation
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
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
