import re


def normalize_num(s: str) -> str:
    """Canonicalize a numeric string: drop thousands commas and trailing-zero decimals.
    '70,000' -> '70000', '18.0' -> '18', '0.50' -> '0.5'."""
    s = s.replace(",", "").strip()
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def extract_answer(text: str) -> str | None:
    """
    Robust GSM8K answer extraction — single source of truth shared by the reward
    function (used for DPO data generation + GRPO rollout scoring) and the evaluator.

    Models write final answers many ways: '#### 18', '#### $70,000', '#### 60%',
    '#### Melanie started with 18 vacuum cleaners.'  We:
      1. If a '####' marker exists, look only at the text after the LAST one and
         take the FIRST number there (the stated final answer).
      2. Otherwise fall back to the LAST number anywhere in the response.
    Commas (thousands separators) are stripped BEFORE number matching so
    '$70,000' -> '70000' instead of splitting into '70' and '000'.
    """
    if "####" in text:
        segment = text.rsplit("####", 1)[-1].replace(",", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", segment)
        if nums:
            return normalize_num(nums[0])
    cleaned = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return normalize_num(nums[-1]) if nums else None


def math_reward(response: str, ground_truth: str) -> float:
    """Binary reward: 1.0 if the response's extracted answer matches ground truth."""
    pred = extract_answer(response)
    if pred is None:
        return 0.0
    return 1.0 if pred == normalize_num(ground_truth) else 0.0
