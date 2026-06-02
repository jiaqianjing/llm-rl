import torch
from algorithms.dpo import dpo_loss


def test_equal_policy_ref_gives_log2():
    # logits = 0 → loss = -log(sigmoid(0)) = log(2) ≈ 0.693
    t = torch.tensor([-1.0])
    loss, cr, rr = dpo_loss(t, t, t, t, beta=0.1)
    assert abs(loss.item() - 0.693) < 0.01


def test_preferred_chosen_gives_lower_loss():
    # policy strongly prefers chosen over rejected relative to ref
    policy_chosen   = torch.tensor([-0.5])
    policy_rejected = torch.tensor([-3.0])
    ref_chosen      = torch.tensor([-1.5])
    ref_rejected    = torch.tensor([-1.5])
    loss_good, _, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    # reversed: policy prefers rejected
    loss_bad, _, _ = dpo_loss(policy_rejected, policy_chosen, ref_chosen, ref_rejected)
    assert loss_good < loss_bad


def test_chosen_reward_higher_than_rejected():
    policy_chosen   = torch.tensor([-0.5])
    policy_rejected = torch.tensor([-2.0])
    ref_chosen      = torch.tensor([-1.0])
    ref_rejected    = torch.tensor([-1.0])
    _, chosen_r, rejected_r = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    assert chosen_r.item() > rejected_r.item()


def test_batch_shape():
    B = 4
    policy_chosen   = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen      = torch.randn(B)
    ref_rejected    = torch.randn(B)
    loss, cr, rr = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    assert loss.shape == ()   # scalar
    assert cr.shape == (B,)
    assert rr.shape == (B,)
