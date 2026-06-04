import re
from datasets import load_dataset


def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\d\.\-,]+)", text)
    if not match:
        return None
    return match.group(1).replace(",", "").strip()


def load_gsm8k(split: str = "train") -> list[dict]:
    """Returns list of {question, answer} dicts."""
    raw = load_dataset("openai/gsm8k", "main", split=split)
    result = []
    for row in raw:
        answer = extract_answer(row["answer"])
        if answer is None:
            continue
        result.append({
            "question": row["question"],
            "answer": answer,
        })
    return result
