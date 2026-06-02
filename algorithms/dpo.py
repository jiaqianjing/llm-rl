import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # [B]
    policy_rejected_logps: torch.Tensor,  # [B]
    ref_chosen_logps: torch.Tensor,       # [B]
    ref_rejected_logps: torch.Tensor,     # [B]
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (loss, chosen_rewards, rejected_rewards)
    chosen_rewards and rejected_rewards are detached scalars per sample.
    """
    # Log ratios relative to reference policy
    logr_chosen   = policy_chosen_logps   - ref_chosen_logps    # [B]
    logr_rejected = policy_rejected_logps - ref_rejected_logps  # [B]

    # DPO objective: maximize margin between chosen and rejected log ratios
    logits = beta * (logr_chosen - logr_rejected)               # [B]
    loss = -F.logsigmoid(logits).mean()

    # For logging: implicit reward estimates (detached)
    chosen_rewards   = (beta * logr_chosen).detach()
    rejected_rewards = (beta * logr_rejected).detach()

    return loss, chosen_rewards, rejected_rewards
