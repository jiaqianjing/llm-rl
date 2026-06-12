from rewards.math_reward import extract_answer, normalize_num, math_reward


def test_normalize_strips_commas_and_trailing_zeros():
    assert normalize_num("70,000") == "70000"
    assert normalize_num("18.0") == "18"
    assert normalize_num("0.50") == "0.5"
    assert normalize_num(" 42 ") == "42"


def test_extract_currency_with_thousands_separator():
    # Regression: "#### $70,000" previously extracted "000" (comma split the number)
    assert extract_answer("#### $70,000") == "70000"
    assert extract_answer("#### $57,500") == "57500"


def test_extract_plain_marker():
    assert extract_answer("#### 18") == "18"
    assert extract_answer("#### $460") == "460"


def test_extract_marker_with_units_and_text():
    assert extract_answer("#### 20 cups") == "20"
    assert extract_answer("#### 60%") == "60"
    assert extract_answer("#### Melanie started with 18 vacuum cleaners.") == "18"
    assert extract_answer("#### The maximum profit is $125.") == "125"


def test_extract_negative():
    assert extract_answer("#### -4") == "-4"


def test_extract_fallback_no_marker():
    assert extract_answer("The answer is 42.") == "42"
    assert extract_answer("no numbers here") is None


def test_math_reward_currency_and_decimal_regressions():
    # Previously mislabeled as WRONG (false negatives) — corrupted DPO preference pairs
    assert math_reward("#### $70,000", "70000") == 1.0
    assert math_reward("#### 18.0", "18") == 1.0          # integer written in decimal form
    assert math_reward("The total is $1,234", "1234") == 1.0  # no marker + thousands comma
    assert math_reward("#### $57,500", "57500") == 1.0
    # genuinely wrong answers stay wrong
    assert math_reward("#### 12", "13") == 0.0
