import pytest

from adaptive_learning_platform.src.bkt import (
    probability_correct,
    update_mastery,
)


PARAMETERS = {
    "initial_mastery": 0.20,
    "learning_probability": 0.15,
    "guess_probability": 0.20,
    "slip_probability": 0.10,
}


def test_initial_probability_correct():
    result = probability_correct(
        prior_mastery=0.20,
        parameters=PARAMETERS,
    )

    assert result == pytest.approx(
        0.34,
        abs=1e-6,
    )


def test_correct_response_increases_mastery():
    result = update_mastery(
        prior_mastery=0.20,
        correct=True,
        parameters=PARAMETERS,
    )

    assert result > 0.20
    assert result == pytest.approx(
        0.60,
        abs=1e-6,
    )


def test_incorrect_response_reduces_mastery():
    result = update_mastery(
        prior_mastery=0.20,
        correct=False,
        parameters=PARAMETERS,
    )

    assert result < 0.20


def test_repeated_correct_answers_increase_mastery():
    mastery = 0.20

    for _ in range(3):
        mastery = update_mastery(
            prior_mastery=mastery,
            correct=True,
            parameters=PARAMETERS,
        )

    assert mastery > 0.90


def test_invalid_parameters_raise_error():
    invalid_parameters = PARAMETERS.copy()
    invalid_parameters[
        "guess_probability"
    ] = 1.50

    with pytest.raises(ValueError):
        probability_correct(
            prior_mastery=0.20,
            parameters=invalid_parameters,
        )