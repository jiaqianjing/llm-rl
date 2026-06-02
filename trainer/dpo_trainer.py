from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from algorithms.dpo import dpo_loss
from algorithms.utils import sequence_logprobs
from trainer.base_trainer import BaseTrainer, TrainerConfig


@dataclass
class DPOConfig(TrainerConfig):
    beta: float = 0.1


class PreferenceDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data  # [{prompt, chosen, rejected}]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DPOTrainer(BaseTrainer):
    def __init__(self, config: DPOConfig, dataset: list[dict]):
        super().__init__(config)
        self.config = config
        ds = PreferenceDataset(dataset)
        self.dataloader = self.accelerator.prepare(
            DataLoader(ds, batch_size=config.batch_size, shuffle=True)
        )

    def _tokenize_pair(self, prompts, responses):
        """Tokenize prompt+response, return input_ids, attention_mask, response_mask."""
        pairs = [p + r for p, r in zip(prompts, responses)]
        enc = self.tokenizer(
            pairs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.accelerator.device)

        prompt_enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.accelerator.device)

        # response_mask: 1 where token belongs to response
        response_mask = torch.zeros_like(enc["attention_mask"], dtype=torch.float)
        for i, prompt_len in enumerate(prompt_enc["attention_mask"].sum(dim=-1)):
            response_mask[i, prompt_len:] = 1.0

        return enc["input_ids"], enc["attention_mask"], response_mask

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

            prompts   = batch["prompt"]
            chosens   = batch["chosen"]
            rejecteds = batch["rejected"]

            # Tokenize chosen and rejected
            chosen_ids, chosen_mask, chosen_resp_mask = self._tokenize_pair(prompts, chosens)
            rejected_ids, rejected_mask, rejected_resp_mask = self._tokenize_pair(prompts, rejecteds)

            # Compute log probs for training model
            policy_chosen_logps = sequence_logprobs(
                self.model, chosen_ids, chosen_mask, chosen_resp_mask
            )
            policy_rejected_logps = sequence_logprobs(
                self.model, rejected_ids, rejected_mask, rejected_resp_mask
            )

            # Compute log probs for ref model (no grad)
            ref_chosen_logps = sequence_logprobs(
                self.ref_model, chosen_ids, chosen_mask, chosen_resp_mask, no_grad=True
            )
            ref_rejected_logps = sequence_logprobs(
                self.ref_model, rejected_ids, rejected_mask, rejected_resp_mask, no_grad=True
            )

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=self.config.beta,
            )

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()

            self.log({
                "loss/train": loss.item(),
                "logr/chosen": chosen_rewards.mean().item(),
                "logr/rejected": rejected_rewards.mean().item(),
                "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            }, step=global_step)

            if global_step % self.config.save_steps == 0 and global_step > 0:
                self.save_checkpoint(global_step)

            global_step += 1
            if global_step % 10 == 0:
                print(f"step {global_step}/{num_steps}  loss={loss.item():.4f}")
