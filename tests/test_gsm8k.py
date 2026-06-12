from data.gsm8k import load_gsm8k, extract_answer


def test_extract_answer():
    text = "Janet sells 16 eggs #### 16"
    assert extract_answer(text) == "16"


def test_extract_answer_missing():
    assert extract_answer("no answer here") is None


def test_load_gsm8k_train():
    ds = load_gsm8k(split="train")
    assert len(ds) > 7000
    row = ds[0]
    assert "question" in row
    assert "answer" in row
    assert row["answer"] is not None


def test_load_gsm8k_test():
    ds = load_gsm8k(split="test")
    assert len(ds) > 1000
