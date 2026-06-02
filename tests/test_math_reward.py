from rewards.math_reward import math_reward


def test_correct_answer():
    assert math_reward("Let me solve. 2x=4 #### 2", "2") == 1.0


def test_wrong_answer():
    assert math_reward("#### 5", "2") == 0.0


def test_no_marker():
    assert math_reward("The answer is two", "2") == 0.0


def test_decimal():
    assert math_reward("#### 3.14", "3.14") == 1.0


def test_whitespace_tolerance():
    assert math_reward("####  42 ", "42") == 1.0


def test_negative():
    assert math_reward("#### -7", "-7") == 1.0
