import torch
from algorithms.utils import sequence_logprobs


def _make_tiny_model():
    """Tiny GPT-2 style model for fast tests."""
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
    return AutoModelForCausalLM.from_config(config).eval()


def test_output_shape():
    model = _make_tiny_model()
    B, L = 2, 10
    input_ids = torch.randint(0, 100, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    response_mask = torch.zeros(B, L, dtype=torch.float)
    response_mask[:, 5:] = 1.0  # last 5 tokens are response

    logps = sequence_logprobs(model, input_ids, attention_mask, response_mask, no_grad=True)
    assert logps.shape == (B,)


def test_logps_are_negative():
    model = _make_tiny_model()
    B, L = 1, 8
    input_ids = torch.randint(0, 100, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    response_mask = torch.ones(B, L, dtype=torch.float)

    logps = sequence_logprobs(model, input_ids, attention_mask, response_mask, no_grad=True)
    assert (logps <= 0).all()


def test_empty_response_gives_zero():
    model = _make_tiny_model()
    B, L = 1, 8
    input_ids = torch.randint(0, 100, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    response_mask = torch.zeros(B, L, dtype=torch.float)  # all prompt, no response

    logps = sequence_logprobs(model, input_ids, attention_mask, response_mask, no_grad=True)
    assert logps[0].item() == 0.0
