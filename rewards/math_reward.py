import re


def math_reward(response: str, ground_truth: str) -> float:
    """Compare response with ground truth. Tries #### format first, then last number."""
    m = re.search(r"####\s*\$?([\d,\.\-]+)", response)
    if m:
        pred = m.group(1).replace(",", "").strip()
        return 1.0 if pred == ground_truth.strip() else 0.0
    # Fallback: last standalone number in the response
    numbers = re.findall(r"(?<![/\d])(-?\d{1,10}(?:\.\d+)?)(?!\d)", response)
    if numbers:
        return 1.0 if numbers[-1] == ground_truth.strip() else 0.0
    return 0.0
