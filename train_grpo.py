import argparse
import yaml

from data.gsm8k import load_gsm8k
from trainer.grpo_trainer import GRPOConfig, GRPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_gsm8k.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_steps = cfg.pop("num_steps", 1000)
    config = GRPOConfig(**cfg)

    dataset = [{"prompt": d["question"], "answer": d["answer"]} for d in load_gsm8k(split="train")]
    trainer = GRPOTrainer(config, dataset)
    trainer.train(num_steps=num_steps)


if __name__ == "__main__":
    main()
