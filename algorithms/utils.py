import contextlib
import torch
import torch.nn.functional as F
from torch import nn


def sequence_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,       # [B, seq_len]
    attention_mask: torch.Tensor,  # [B, seq_len]
    response_mask: torch.Tensor,   # [B, seq_len], 1.0 for response tokens
    no_grad: bool = False,
) -> torch.Tensor:                 # [B] sum of log probs over response tokens
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]              # [B, L-1, vocab]
    targets = input_ids[:, 1:]                      # [B, L-1]
    mask = response_mask[:, 1:].float()             # [B, L-1], shift to align with targets
    # Avoid materialising [B, L-1, vocab] log_softmax (~10 GB for batch=32, vocab=152k).
    # gather target logit then subtract logsumexp — numerically identical, O(B*L) memory.
    token_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    log_Z = torch.logsumexp(logits, dim=-1)                              # [B, L-1]
    token_logps = token_logits - log_Z                                   # [B, L-1]
    return (token_logps * mask).sum(dim=-1)         # [B]
