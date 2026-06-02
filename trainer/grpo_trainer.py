from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from algorithms.grpo import grpo_loss, compute_advantages
from algorithms.utils import sequence_logprobs
from rewards.math_reward import math_reward
from trainer.base_trainer import BaseTrainer, TrainerConfig


@dataclass
class GRPOConfig(TrainerConfig):
    beta: float = 0.04
    group_size: int = 8           # G: responses per prompt
    temperature: float = 0.8
    reward_fn_name: str = "math"  # "math" only in Phase 1


class PromptDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data  # [{prompt, answer}]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GRPOTrainer(BaseTrainer):
    def __init__(self, config: GRPOConfig, dataset: list[dict]):
        super().__init__(config)
        self.config = config
        ds = PromptDataset(dataset)
        self.dataloader = self.accelerator.prepare(
            DataLoader(ds, batch_size=config.batch_size, shuffle=True)
        )

    def _rollout(self, prompts: list[str]) -> list[list[str]]:
        """Generate G responses per prompt using model.generate()."""
        G = self.config.group_size
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.accelerator.device)

        unwrapped = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            output_ids = unwrapped.generate(
                **enc,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=G,
                do_sample=True,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # output_ids: [B*G, seq_len]
        prompt_len = enc["input_ids"].shape[-1]
        responses_flat = [
            self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            for ids in output_ids
        ]
        # Reshape to [B, G]
        B = len(prompts)
        return [responses_flat[i * G:(i + 1) * G] for i in range(B)]

    def _compute_rewards(
        self, prompts: list[str], responses: list[list[str]], answers: list[str]
    ) -> torch.Tensor:
        """Compute scalar reward per (prompt, response). Returns [B, G]."""
        B, G = len(prompts), self.config.group_size
        rewards = torch.zeros(B, G, device=self.accelerator.device)
        for i, (resps, ans) in enumerate(zip(responses, answers)):
            for j, resp in enumerate(resps):
                rewards[i, j] = math_reward(resp, ans)
        return rewards

    def _compute_logprobs(
        self, prompts: list[str], responses: list[list[str]], no_grad: bool = False
    ) -> torch.Tensor:
        """Compute sequence log probs. Returns [B, G]."""
        G = self.config.group_size
        # Flatten [B*G] pairs for batched forward pass
        flat_pairs = [
            p + r
            for p, resps in zip(prompts, responses)
            for r in resps
        ]
        flat_prompts = [p for p in prompts for _ in range(G)]
        enc = self.tokenizer(
            flat_pairs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.accelerator.device)
        prompt_enc = self.tokenizer(
            flat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.accelerator.device)

        # Build response mask
        response_mask = torch.zeros_like(enc["attention_mask"], dtype=torch.float)
        for i, plen in enumerate(prompt_enc["attention_mask"].sum(dim=-1)):
            response_mask[i, plen:] = 1.0

        model = self.ref_model if no_grad else self.model
        flat_logps = sequence_logprobs(
            model, enc["input_ids"], enc["attention_mask"], response_mask, no_grad=no_grad
        )  # [B*G]
        return flat_logps.view(len(prompts), G)   # [B, G]

    def train(self, num_steps: int):
        self.model.train()
        data_iter: Iterator = iter(self.dataloader)
        global_step = 0

        while global_step < num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            prompts = batch["prompt"]
            answers = batch["answer"]

            # ① Rollout: generate G responses per prompt
            responses = self._rollout(list(prompts))

            # ② Score with reward function
            rewards = self._compute_rewards(list(prompts), responses, list(answers))  # [B, G]

            # ③ Compute log probs (training model + ref model)
            log_probs = self._compute_logprobs(list(prompts), responses, no_grad=False)  # [B, G]
            ref_log_probs = self._compute_logprobs(list(prompts), responses, no_grad=True)

            # ④ GRPO loss
            loss, advantages = grpo_loss(log_probs, ref_log_probs, rewards, beta=self.config.beta)

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()

            self.log({
                "loss/train": loss.item(),
                "reward/mean": rewards.mean().item(),
                "reward/std": rewards.std().item(),
                "reward/pass_rate": (rewards == 1.0).float().mean().item(),
                "advantage/mean": advantages.mean().item(),
                "kl/mean": (log_probs - ref_log_probs).mean().item(),
            }, step=global_step)

            if global_step % self.config.save_steps == 0 and global_step > 0:
                self.save_checkpoint(global_step)

            global_step += 1
            if global_step % 10 == 0:
                print(
                    f"step {global_step}/{num_steps}  "
                    f"loss={loss.item():.4f}  "
                    f"reward={rewards.mean().item():.3f}"
                )
