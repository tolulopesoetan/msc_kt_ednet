from pathlib import Path

import pandas as pd

from .config import (
    ALLOWED_DIFFICULTIES,
    ITEM_REGISTER_PATH,
    ITEM_REGISTER_SHEET,
    QUESTION_BANK_PATH,
    REQUIRED_QUESTION_COLUMNS,
    SKILL_NAMES,
)


def load_question_bank(
    question_bank_path=QUESTION_BANK_PATH,
    item_register_path=ITEM_REGISTER_PATH,
    item_register_sheet=ITEM_REGISTER_SHEET,
):
    question_bank_path = Path(
        question_bank_path
    )

    item_register_path = Path(
        item_register_path
    )

    if not question_bank_path.exists():
        raise FileNotFoundError(
            "Question bank not found: "
            f"{question_bank_path}"
        )

    if not item_register_path.exists():
        raise FileNotFoundError(
            "SFLA item register not found: "
            f"{item_register_path}"
        )

    question_bank = pd.read_csv(
        question_bank_path
    )

    item_register = pd.read_excel(
        item_register_path,
        sheet_name=item_register_sheet,
    )

    missing_columns = (
        REQUIRED_QUESTION_COLUMNS
        - set(question_bank.columns)
    )

    if missing_columns:
        raise ValueError(
            "Question bank is missing columns: "
            f"{sorted(missing_columns)}"
        )

    required_values = question_bank[
        sorted(REQUIRED_QUESTION_COLUMNS)
    ]

    if required_values.isna().any().any():
        columns_with_missing_values = (
            required_values.columns[
                required_values.isna().any()
            ].tolist()
        )

        raise ValueError(
            "Missing question-bank values in: "
            f"{columns_with_missing_values}"
        )

    question_bank["item_id"] = (
        question_bank["item_id"]
        .astype(str)
        .str.strip()
    )

    question_bank["skill_id"] = (
        question_bank["skill_id"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    question_bank["difficulty"] = (
        question_bank["difficulty"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    question_bank["correct_option"] = (
        question_bank["correct_option"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if question_bank["item_id"].duplicated().any():
        duplicated_ids = question_bank.loc[
            question_bank["item_id"].duplicated(
                keep=False
            ),
            "item_id",
        ].tolist()

        raise ValueError(
            "Duplicate question IDs: "
            f"{sorted(set(duplicated_ids))}"
        )

    invalid_answers = (
        set(question_bank["correct_option"])
        - {"A", "B", "C", "D"}
    )

    if invalid_answers:
        raise ValueError(
            "Invalid correct-option values: "
            f"{sorted(invalid_answers)}"
        )

    invalid_difficulties = (
        set(question_bank["difficulty"])
        - ALLOWED_DIFFICULTIES
    )

    if invalid_difficulties:
        raise ValueError(
            "Invalid difficulty values: "
            f"{sorted(invalid_difficulties)}"
        )

    registered_skills = set(
        item_register["skill_id"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
    )

    invalid_skills = (
        set(question_bank["skill_id"])
        - registered_skills
    )

    if invalid_skills:
        raise ValueError(
            "Question-bank skills are not present "
            "in the SFLA register: "
            f"{sorted(invalid_skills)}"
        )

    missing_skills = (
        set(SKILL_NAMES)
        - set(question_bank["skill_id"])
    )

    if missing_skills:
        raise ValueError(
            "The question bank does not cover: "
            f"{sorted(missing_skills)}"
        )

    for skill_id in SKILL_NAMES:
        skill_difficulties = set(
            question_bank.loc[
                question_bank["skill_id"]
                == skill_id,
                "difficulty",
            ]
        )

        missing_difficulties = (
            ALLOWED_DIFFICULTIES
            - skill_difficulties
        )

        if missing_difficulties:
            raise ValueError(
                f"{skill_id} is missing: "
                f"{sorted(missing_difficulties)}"
            )

    return question_bank