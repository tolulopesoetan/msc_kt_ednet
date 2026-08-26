import pandas as pd
import pytest

from adaptive_learning_platform.src.recommender import (
    recommended_difficulty,
    select_next_question,
)


@pytest.mark.parametrize(
    "mastery, expected_difficulty",
    [
        (0.20, "easy"),
        (0.39, "easy"),
        (0.40, "medium"),
        (0.69, "medium"),
        (0.70, "hard"),
        (0.95, "hard"),
    ],
)
def test_recommended_difficulty(
    mastery,
    expected_difficulty,
):
    result = recommended_difficulty(mastery)

    assert result == expected_difficulty


def test_weakest_skill_is_selected():
    question_bank = pd.DataFrame(
        [
            {
                "item_id": "KC01_E",
                "skill_id": "KC01",
                "difficulty": "easy",
            },
            {
                "item_id": "KC02_E",
                "skill_id": "KC02",
                "difficulty": "easy",
            },
        ]
    )

    mastery = {
        "KC01": 0.80,
        "KC02": 0.20,
    }

    result = select_next_question(
        question_bank=question_bank,
        mastery=mastery,
        attempted_item_ids=set(),
        random_state=42,
    )

    assert result["skill_id"] == "KC02"
    assert result["item_id"] == "KC02_E"


def test_difficulty_matches_mastery():
    question_bank = pd.DataFrame(
        [
            {
                "item_id": "KC01_E",
                "skill_id": "KC01",
                "difficulty": "easy",
            },
            {
                "item_id": "KC01_M",
                "skill_id": "KC01",
                "difficulty": "medium",
            },
            {
                "item_id": "KC01_H",
                "skill_id": "KC01",
                "difficulty": "hard",
            },
        ]
    )

    mastery = {
        "KC01": 0.50,
    }

    result = select_next_question(
        question_bank=question_bank,
        mastery=mastery,
        attempted_item_ids=set(),
        random_state=42,
    )

    assert result["difficulty"] == "medium"
    assert result["item_id"] == "KC01_M"


def test_attempted_question_is_not_selected():
    question_bank = pd.DataFrame(
        [
            {
                "item_id": "KC01_E",
                "skill_id": "KC01",
                "difficulty": "easy",
            },
            {
                "item_id": "KC01_M",
                "skill_id": "KC01",
                "difficulty": "medium",
            },
        ]
    )

    mastery = {
        "KC01": 0.20,
    }

    result = select_next_question(
        question_bank=question_bank,
        mastery=mastery,
        attempted_item_ids={"KC01_E"},
        random_state=42,
    )

    assert result["item_id"] == "KC01_M"


def test_none_returned_when_all_questions_attempted():
    question_bank = pd.DataFrame(
        [
            {
                "item_id": "KC01_E",
                "skill_id": "KC01",
                "difficulty": "easy",
            }
        ]
    )

    mastery = {
        "KC01": 0.20,
    }

    result = select_next_question(
        question_bank=question_bank,
        mastery=mastery,
        attempted_item_ids={"KC01_E"},
        random_state=42,
    )

    assert result is None