import torch
from algorithms.grpo import grpo_loss, compute_advantages


def test_advantages_zero_when_all_same_reward():
    rewards = torch.ones(2, 4)
    advantages = compute_advantages(rewards)
    assert (advantages.abs() < 1e-6).all()


def test_advantages_normalized():
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    advantages = compute_advantages(rewards)
    assert abs(advantages.mean().item()) < 1e-5
    assert abs(advantages.std().item() - 1.0) < 0.01


def test_positive_advantage_for_higher_reward():
    rewards = torch.tensor([[1.0, 0.0]])
    advantages = compute_advantages(rewards)
    assert advantages[0, 0] > 0   # reward=1 → positive advantage
    assert advantages[0, 1] < 0   # reward=0 → negative advantage


def test_grpo_loss_shape():
    B, G = 2, 4
    log_probs = torch.randn(B, G)
    ref_log_probs = torch.randn(B, G)
    rewards = torch.rand(B, G)
    loss, advantages = grpo_loss(log_probs, ref_log_probs, rewards)
    assert loss.shape == ()          # scalar
    assert advantages.shape == (B, G)


def test_kl_beta_zero_ignores_ref():
    B, G = 1, 4
    log_probs = torch.tensor([[-0.5, -2.0, -0.3, -1.8]])
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    ref_same = log_probs.clone()
    ref_diff = torch.full((B, G), -5.0)

    loss_same, _ = grpo_loss(log_probs, ref_same, rewards, beta=0.0)
    loss_diff, _ = grpo_loss(log_probs, ref_diff, rewards, beta=0.0)
    assert abs(loss_same.item() - loss_diff.item()) < 1e-6
