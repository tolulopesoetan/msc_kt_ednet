import pandas as pd
import pytest

from adaptive_learning_platform.src.config import (
    ALLOWED_DIFFICULTIES,
    ITEM_REGISTER_SHEET,
    SKILL_NAMES,
)
from adaptive_learning_platform.src.data_loader import (
    load_question_bank,
)


def create_valid_question_bank():
    questions = []
    item_number = 1

    for skill_id in SKILL_NAMES:
        for difficulty in sorted(
            ALLOWED_DIFFICULTIES
        ):
            questions.append(
                {
                    "item_id": (
                        f"TEST_{item_number:03d}"
                    ),
                    "skill_id": skill_id,
                    "difficulty": difficulty,
                    "question_text": (
                        f"Test question for "
                        f"{skill_id}."
                    ),
                    "option_a": "Option A",
                    "option_b": "Option B",
                    "option_c": "Option C",
                    "option_d": "Option D",
                    "correct_option": "A",
                    "explanation": (
                        "Test explanation."
                    ),
                }
            )

            item_number += 1

    return pd.DataFrame(questions)


def write_test_files(
    tmp_path,
    question_bank,
):
    question_bank_path = (
        tmp_path / "question_bank.csv"
    )

    item_register_path = (
        tmp_path / "item_register.xlsx"
    )

    question_bank.to_csv(
        question_bank_path,
        index=False,
    )

    item_register = pd.DataFrame(
        {
            "skill_id": list(
                SKILL_NAMES.keys()
            )
        }
    )

    item_register.to_excel(
        item_register_path,
        sheet_name=ITEM_REGISTER_SHEET,
        index=False,
    )

    return (
        question_bank_path,
        item_register_path,
    )


def test_valid_question_bank_passes(
    tmp_path,
):
    question_bank = (
        create_valid_question_bank()
    )

    (
        question_bank_path,
        item_register_path,
    ) = write_test_files(
        tmp_path,
        question_bank,
    )

    loaded_question_bank = (
        load_question_bank(
            question_bank_path=(
                question_bank_path
            ),
            item_register_path=(
                item_register_path
            ),
        )
    )

    assert len(loaded_question_bank) == 27

    assert (
        loaded_question_bank["skill_id"]
        .nunique()
        == 9
    )


def test_missing_column_is_rejected(
    tmp_path,
):
    question_bank = (
        create_valid_question_bank()
        .drop(columns=["explanation"])
    )

    (
        question_bank_path,
        item_register_path,
    ) = write_test_files(
        tmp_path,
        question_bank,
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        load_question_bank(
            question_bank_path=(
                question_bank_path
            ),
            item_register_path=(
                item_register_path
            ),
        )


def test_duplicate_item_id_is_rejected(
    tmp_path,
):
    question_bank = (
        create_valid_question_bank()
    )

    question_bank.loc[
        1,
        "item_id",
    ] = question_bank.loc[
        0,
        "item_id",
    ]

    (
        question_bank_path,
        item_register_path,
    ) = write_test_files(
        tmp_path,
        question_bank,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate question IDs",
    ):
        load_question_bank(
            question_bank_path=(
                question_bank_path
            ),
            item_register_path=(
                item_register_path
            ),
        )


def test_unregistered_skill_is_rejected(
    tmp_path,
):
    question_bank = (
        create_valid_question_bank()
    )

    question_bank.loc[
        0,
        "skill_id",
    ] = "KC99"

    (
        question_bank_path,
        item_register_path,
    ) = write_test_files(
        tmp_path,
        question_bank,
    )

    with pytest.raises(
        ValueError,
        match="not present",
    ):
        load_question_bank(
            question_bank_path=(
                question_bank_path
            ),
            item_register_path=(
                item_register_path
            ),
        )