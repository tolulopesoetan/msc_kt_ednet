from pathlib import Path


APP_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIRECTORY.parent

QUESTION_BANK_PATH = (
    PROJECT_ROOT
    / "data"
    / "platform"
    / "sfla_question_bank.csv"
)

ITEM_REGISTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "SFLA_Item_Register_v1.xlsx"
)

DATABASE_PATH = (
    APP_DIRECTORY
    / "storage"
    / "sfla_learning.db"
)

ITEM_REGISTER_SHEET = "sfla_item_skill_map_v1"


SKILL_NAMES = {
    "KC01": "Logical equivalence and truth tables",
    "KC02": "Quantifiers and statement transformations",
    "KC03": "Set notation and membership",
    "KC04": "Set operations",
    "KC05": "Power sets and Cartesian products",
    "KC06": "Set-based proof",
    "KC07": "Mathematical induction",
    "KC08": "Function properties",
    "KC09": "Inverse and composite functions",
}


BKT_PARAMETERS = {
    "initial_mastery": 0.20,
    "learning_probability": 0.15,
    "guess_probability": 0.20,
    "slip_probability": 0.10,
}


REQUIRED_QUESTION_COLUMNS = {
    "item_id",
    "skill_id",
    "difficulty",
    "question_text",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "correct_option",
    "explanation",
}


ALLOWED_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}