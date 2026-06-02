import argparse
import json
import yaml

from trainer.dpo_trainer import DPOConfig, DPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_gsm8k.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = cfg.pop("data_path")
    num_steps = cfg.pop("num_steps", 1000)
    config = DPOConfig(**cfg)

    with open(data_path) as f:
        dataset = json.load(f)

    trainer = DPOTrainer(config, dataset)
    trainer.train(num_steps=num_steps)


if __name__ == "__main__":
    main()
