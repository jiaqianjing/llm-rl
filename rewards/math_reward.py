import re


def math_reward(response: str, ground_truth: str) -> float:
    """Extract answer after #### and compare with ground truth."""
    match = re.search(r"####\s*([\d\.\-]+)", response)
    if not match:
        return 0.0
    return 1.0 if match.group(1).strip() == ground_truth.strip() else 0.0
