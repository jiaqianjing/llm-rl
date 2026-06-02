import torch


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Group-relative advantage normalization.
    rewards: [B, G]
    returns: [B, G], normalized advantages (mean=0, std=1 per group)
    """
    mean_r = rewards.mean(dim=-1, keepdim=True)             # [B, 1]
    std_r  = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return (rewards - mean_r) / std_r                       # [B, G]


def grpo_loss(
    log_probs: torch.Tensor,       # [B, G] sequence-level sum of log probs
    ref_log_probs: torch.Tensor,   # [B, G]
    rewards: torch.Tensor,         # [B, G]
    beta: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (total_loss, advantages)
    total_loss = policy_gradient_loss + beta * kl_penalty
    """
    advantages = compute_advantages(rewards)                 # [B, G]

    # Policy gradient: maximize advantage-weighted log prob
    pg_loss = -(advantages * log_probs).mean()

    # KL penalty: stay close to reference policy
    kl = (log_probs - ref_log_probs).mean()

    total_loss = pg_loss + beta * kl
    return total_loss, advantages
