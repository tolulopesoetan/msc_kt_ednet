from pathlib import Path

import pandas as pd


QUESTIONS = [
    {
        "item_id": "SFLA_APP_001",
        "skill_id": "KC01",
        "difficulty": "easy",
        "question_text": "Which expression is logically equivalent to P → Q?",
        "option_a": "P ∨ Q",
        "option_b": "P ∧ Q",
        "option_c": "¬P ∨ Q",
        "option_d": "P ∨ ¬Q",
        "correct_option": "C",
        "explanation": (
            "An implication P → Q is logically equivalent to ¬P ∨ Q."
        ),
        "source": "Original prototype practice question",
    },
    {
        "item_id": "SFLA_APP_002",
        "skill_id": "KC02",
        "difficulty": "medium",
        "question_text": "What is the negation of ∀x P(x)?",
        "option_a": "∀x ¬P(x)",
        "option_b": "∃x ¬P(x)",
        "option_c": "∃x P(x)",
        "option_d": "¬∃x P(x)",
        "correct_option": "B",
        "explanation": (
            "The negation of 'P is true for every x' is "
            "'there exists an x for which P is false'."
        ),
        "source": "Original prototype practice question",
    },
    {
        "item_id": "SFLA_APP_003",
        "skill_id": "KC03",
        "difficulty": "easy",
        "question_text": (
            "Which symbol means that an object is an element of a set?"
        ),
        "option_a": "⊆",
        "option_b": "∈",
        "option_c": "∩",
        "option_d": "∪",
        "correct_option": "B",
        "explanation": (
            "The symbol ∈ denotes membership of a set."
        ),
        "source": "Original prototype practice question",
    },
    {
        "item_id": "SFLA_APP_004",
        "skill_id": "KC04",
        "difficulty": "medium",
        "question_text": (
            "If A = {1, 2, 3} and B = {3, 4, 5}, what is A ∩ B?"
        ),
        "option_a": "{1, 2}",
        "option_b": "{3}",
        "option_c": "{4, 5}",
        "option_d": "{1, 2, 3, 4, 5}",
        "correct_option": "B",
        "explanation": (
            "The intersection contains elements belonging to both sets."
        ),
        "source": "Original prototype practice question",
    },
    {
        "item_id": "SFLA_APP_005",
        "skill_id": "KC05",
        "difficulty": "medium",
        "question_text": (
            "How many elements are in the power set of a "
            "set containing three elements?"
        ),
        "option_a": "3",
        "option_b": "6",
        "option_c": "8",
        "option_d": "9",
        "correct_option": "C",
        "explanation": (
            "A set with n elements has 2ⁿ subsets. Therefore, 2³ = 8."
        ),
        "source": "Original prototype practice question",
    },
    {
        "item_id": "SFLA_APP_006",
        "skill_id": "KC06",
        "difficulty": "medium",
        "question_text": (
            "What is normally required to prove that two sets A and B "
            "are equal?"
        ),
        "option_a": "Show only that A ⊆ B",
        "option_b": "Show only that B ⊆ A",
        "option_c": "Show that A and B have the same notation",
        "option_d": "Show that A ⊆ B and B ⊆ A",
        "correct_option": "D",
        "explanation": (
            "Set equality can be proved by establishing inclusion "
            "in both directions."
        ),
        "source": "Original prototype practice question",
    },
    {
        "item_id": "SFLA_APP_007",
        "skill_id": "KC07",
        "difficulty": "medium",
        "question_text": (
            "During the inductive step, what is assumed before proving "
            "P(k + 1)?"
        ),
        "option_a": "P(0) is false",
        "option_b": "P(k) is true",
        "option_c": "P(k + 1) is false",
        "option_d": "Every statement is true",
        "correct_option": "B",
        "explanation": (
            "The inductive hypothesis assumes P(k) is true and uses "
            "that assumption to prove P(k + 1)."
        ),
        "source": "Original prototype practice question",
    },
    {
        "item_id": "SFLA_APP_008",
        "skill_id": "KC08",
        "difficulty": "medium",
        "question_text": "When is a function f injective?",
        "option_a": (
            "When every element of the codomain has at least one preimage"
        ),
        "option_b": "When different inputs always have different outputs",
        "option_c": "When every input has two outputs",
        "option_d": "When the domain and codomain are empty",
        "correct_option": "B",
        "explanation": (
            "An injective function does not map two distinct inputs "
            "to the same output."
        ),
        "source": "Original prototype practice question",
    },
    {
        "item_id": "SFLA_APP_009",
        "skill_id": "KC09",
        "difficulty": "easy",
        "question_text": "What does (f ∘ g)(x) mean?",
        "option_a": "g(f(x))",
        "option_b": "f(x) + g(x)",
        "option_c": "f(g(x))",
        "option_d": "f(x)g(x)",
        "correct_option": "C",
        "explanation": (
            "In the composition f ∘ g, g is applied first and f is "
            "then applied to the result."
        ),
        "source": "Original prototype practice question",
    },
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "platform"
    / "sfla_question_bank.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

question_bank = pd.DataFrame(QUESTIONS)
question_bank.to_csv(OUTPUT_PATH, index=False)

print(f"Created {len(question_bank)} questions.")
print(f"Question bank saved to: {OUTPUT_PATH}")