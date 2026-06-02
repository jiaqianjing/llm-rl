# llm-rl

从零手写 GRPO 与 DPO，在 Qwen2.5-7B-Instruct 上对比两种算法的训练动态与效果。

## 目标

- 裸写 GRPO 核心逻辑：group sampling、advantage 计算、policy gradient
- 裸写 DPO 核心逻辑：log ratio contrastive loss
- Phase 1：GSM8K 数学推理（verifiable reward）
- Phase 2：UltraFeedback 指令跟随（LLM judge reward）
- 用 oxRL 做 sanity check 对照

## 硬件

4x A100 80G（GPU 4-7），Accelerate FSDP ZeRO-3

## 文档

- [GRPO vs DPO 概念详解](docs/grpo-vs-dpo-concepts.md)
- [RL 本质与 Meta-Learning](docs/rl-insights-and-meta-learning.md)
- [数据流与分布式方案](docs/data-pipeline-and-distributed.md)
- [设计文档](docs/superpowers/specs/2026-06-02-llm-rl-design.md)

## 参考

- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948)（GRPO 来源）
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [oxRL](https://github.com/warlockee/oxRL)（对照框架）
