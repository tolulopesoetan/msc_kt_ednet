from pathlib import Path

import pandas as pd


def question(
    item_id,
    skill_id,
    difficulty,
    question_text,
    option_a,
    option_b,
    option_c,
    option_d,
    correct_option,
    explanation,
):
    return {
        "item_id": item_id,
        "skill_id": skill_id,
        "difficulty": difficulty,
        "question_text": question_text,
        "option_a": option_a,
        "option_b": option_b,
        "option_c": option_c,
        "option_d": option_d,
        "correct_option": correct_option,
        "explanation": explanation,
        "source": "Original prototype practice question",
    }


QUESTIONS = [
    # KC01: Logical equivalence and truth tables
    question(
        "SFLA_APP_001",
        "KC01",
        "easy",
        "Which expression is logically equivalent to P → Q?",
        "P ∨ Q",
        "P ∧ Q",
        "¬P ∨ Q",
        "P ∨ ¬Q",
        "C",
        "An implication P → Q is equivalent to ¬P ∨ Q.",
    ),
    question(
        "SFLA_APP_002",
        "KC01",
        "medium",
        "Using De Morgan's law, what is the negation of P ∧ Q?",
        "¬P ∧ ¬Q",
        "¬P ∨ ¬Q",
        "P ∨ Q",
        "P → Q",
        "B",
        "The negation of a conjunction is the disjunction "
        "of the individual negations.",
    ),
    question(
        "SFLA_APP_003",
        "KC01",
        "hard",
        "In which case is the implication P → Q false?",
        "P is true and Q is true",
        "P is true and Q is false",
        "P is false and Q is true",
        "P is false and Q is false",
        "B",
        "An implication is false only when its antecedent is "
        "true and its consequent is false.",
    ),

    # KC02: Quantifiers and statement transformations
    question(
        "SFLA_APP_004",
        "KC02",
        "easy",
        "What is the negation of ∀x P(x)?",
        "∀x ¬P(x)",
        "∃x ¬P(x)",
        "∃x P(x)",
        "¬∃x ¬P(x)",
        "B",
        "The negation states that there is at least one x "
        "for which P(x) is false.",
    ),
    question(
        "SFLA_APP_005",
        "KC02",
        "medium",
        "What is the negation of ∃x P(x)?",
        "∀x ¬P(x)",
        "∃x ¬P(x)",
        "∀x P(x)",
        "¬∀x ¬P(x)",
        "A",
        "If no x satisfies P, then P is false for every x.",
    ),
    question(
        "SFLA_APP_006",
        "KC02",
        "hard",
        "Which expression means 'every learner attempted at "
        "least one question'?",
        "∃l ∀q Attempted(l, q)",
        "∀l ∃q Attempted(l, q)",
        "∀q ∃l Attempted(l, q)",
        "∃q ∀l Attempted(l, q)",
        "B",
        "For every learner l, there must exist at least one "
        "question q that the learner attempted.",
    ),

    # KC03: Set notation and membership
    question(
        "SFLA_APP_007",
        "KC03",
        "easy",
        "Which symbol means that an object is an element of a set?",
        "⊆",
        "∈",
        "∩",
        "∪",
        "B",
        "The symbol ∈ denotes membership of a set.",
    ),
    question(
        "SFLA_APP_008",
        "KC03",
        "medium",
        "Let A = {1, {2}, 3}. Which statement is true?",
        "2 ∈ A",
        "{2} ∈ A",
        "{1} ∈ A",
        "{2} ⊆ A",
        "B",
        "The set {2} is itself listed as an element of A. "
        "The number 2 is not listed separately.",
    ),
    question(
        "SFLA_APP_009",
        "KC03",
        "hard",
        "Which set is described by {x ∈ ℤ | x² < 5}?",
        "{−1, 0, 1}",
        "{−2, −1, 0, 1, 2}",
        "{0, 1, 2}",
        "{−4, −1, 0, 1, 4}",
        "B",
        "The integers whose squares are less than 5 are "
        "−2, −1, 0, 1 and 2.",
    ),

    # KC04: Set operations
    question(
        "SFLA_APP_010",
        "KC04",
        "easy",
        "If A = {1, 2, 3} and B = {3, 4, 5}, what is A ∩ B?",
        "{1, 2}",
        "{3}",
        "{4, 5}",
        "{1, 2, 3, 4, 5}",
        "B",
        "The intersection contains elements found in both sets.",
    ),
    question(
        "SFLA_APP_011",
        "KC04",
        "medium",
        "If A = {1, 2, 3, 4} and B = {3, 4, 5}, what is A \\ B?",
        "{1, 2}",
        "{3, 4}",
        "{5}",
        "{1, 2, 5}",
        "A",
        "A \\ B contains elements that belong to A but not to B.",
    ),
    question(
        "SFLA_APP_012",
        "KC04",
        "hard",
        "Which expression is equivalent to (A ∪ B)ᶜ?",
        "Aᶜ ∪ Bᶜ",
        "Aᶜ ∩ Bᶜ",
        "A ∩ B",
        "A ∪ B",
        "B",
        "De Morgan's law gives (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ.",
    ),

    # KC05: Power sets and Cartesian products
    question(
        "SFLA_APP_013",
        "KC05",
        "easy",
        "How many elements are in the power set of a set "
        "containing three elements?",
        "3",
        "6",
        "8",
        "9",
        "C",
        "A set with n elements has 2ⁿ subsets, so 2³ = 8.",
    ),
    question(
        "SFLA_APP_014",
        "KC05",
        "medium",
        "If |A| = 2 and |B| = 3, what is |A × B|?",
        "2",
        "3",
        "5",
        "6",
        "D",
        "The Cartesian product contains |A| × |B| ordered pairs.",
    ),
    question(
        "SFLA_APP_015",
        "KC05",
        "hard",
        "Which is the power set of {a, b}?",
        "{a, b}",
        "{∅, {a}, {b}, {a, b}}",
        "{{a}, {b}}",
        "{∅, a, b}",
        "B",
        "The power set contains every subset: the empty set, "
        "both singleton sets and the complete set.",
    ),

    # KC06: Set-based proof
    question(
        "SFLA_APP_016",
        "KC06",
        "easy",
        "What must normally be shown to prove that A = B?",
        "Only A ⊆ B",
        "Only B ⊆ A",
        "A ⊆ B and B ⊆ A",
        "A and B use the same notation",
        "C",
        "Set equality can be proved by inclusion in both directions.",
    ),
    question(
        "SFLA_APP_017",
        "KC06",
        "medium",
        "What is an appropriate first step when proving A ⊆ B?",
        "Choose an arbitrary x ∈ A",
        "Assume that A = B",
        "Choose an x that is not in A",
        "Prove that B is empty",
        "A",
        "Select an arbitrary element of A and demonstrate that "
        "it must also belong to B.",
    ),
    question(
        "SFLA_APP_018",
        "KC06",
        "hard",
        "Which statement supports the proof of "
        "A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)?",
        "x ∈ A or x ∈ B and x ∈ C",
        "x ∈ A and (x ∈ B or x ∈ C)",
        "x ∉ A and x ∈ B",
        "x ∈ A only",
        "B",
        "Distributing 'and' over 'or' gives "
        "(x ∈ A and x ∈ B) or (x ∈ A and x ∈ C).",
    ),

    # KC07: Mathematical induction
    question(
        "SFLA_APP_019",
        "KC07",
        "easy",
        "What is the first stage of a proof by mathematical induction?",
        "Assume the conclusion",
        "Verify the base case",
        "Prove the statement is false",
        "Choose a counterexample",
        "B",
        "The base case establishes that the statement is true "
        "for the initial value.",
    ),
    question(
        "SFLA_APP_020",
        "KC07",
        "medium",
        "During the inductive step, what is assumed before "
        "proving P(k + 1)?",
        "P(k) is false",
        "P(k) is true",
        "P(k + 1) is false",
        "Every possible statement is true",
        "B",
        "P(k) is assumed to be true as the inductive hypothesis.",
    ),
    question(
        "SFLA_APP_021",
        "KC07",
        "hard",
        "For the induction proof of 1 + 2 + ... + n = n(n + 1)/2, "
        "which expression is obtained for k + 1?",
        "k(k + 1)/2 + (k + 1)",
        "k(k + 1)/2 + k",
        "(k + 1)(k + 1)/2",
        "k(k + 2)/2",
        "A",
        "The next term k + 1 is added to the expression supplied "
        "by the inductive hypothesis.",
    ),

    # KC08: Function properties
    question(
        "SFLA_APP_022",
        "KC08",
        "easy",
        "When is a function injective?",
        "Every output has at least one input",
        "Different inputs always produce different outputs",
        "Every input produces two outputs",
        "The domain equals the codomain",
        "B",
        "An injective function does not map distinct inputs "
        "to the same output.",
    ),
    question(
        "SFLA_APP_023",
        "KC08",
        "medium",
        "When is a function surjective?",
        "Every codomain element has at least one preimage",
        "Every domain element has two images",
        "Different inputs always have different outputs",
        "The function has no domain",
        "A",
        "A surjective function reaches every element of its codomain.",
    ),
    question(
        "SFLA_APP_024",
        "KC08",
        "hard",
        "For f: ℝ → ℝ defined by f(x) = x², which statement is correct?",
        "f is injective and surjective",
        "f is injective but not surjective",
        "f is surjective but not injective",
        "f is neither injective nor surjective",
        "D",
        "Different inputs such as 2 and −2 have the same output, "
        "and negative real numbers are not reached.",
    ),

    # KC09: Inverse and composite functions
    question(
        "SFLA_APP_025",
        "KC09",
        "easy",
        "What does (f ∘ g)(x) mean?",
        "g(f(x))",
        "f(x) + g(x)",
        "f(g(x))",
        "f(x)g(x)",
        "C",
        "The function g is applied first, followed by f.",
    ),
    question(
        "SFLA_APP_026",
        "KC09",
        "medium",
        "If f(x) = 2x + 3, what is f⁻¹(x)?",
        "(x + 3)/2",
        "(x − 3)/2",
        "2x − 3",
        "1/(2x + 3)",
        "B",
        "Set y = 2x + 3 and rearrange to obtain x = (y − 3)/2.",
    ),
    question(
        "SFLA_APP_027",
        "KC09",
        "hard",
        "If f and g are invertible, which expression equals "
        "(f ∘ g)⁻¹?",
        "f⁻¹ ∘ g⁻¹",
        "g⁻¹ ∘ f⁻¹",
        "f ∘ g",
        "g ∘ f",
        "B",
        "The order is reversed when a composition is inverted.",
    ),
]


question_bank = pd.DataFrame(QUESTIONS)

expected_skills = {
    f"KC{number:02d}"
    for number in range(1, 10)
}

expected_difficulties = {
    "easy",
    "medium",
    "hard",
}

if question_bank["item_id"].duplicated().any():
    raise ValueError("Duplicate question IDs were detected.")

if set(question_bank["skill_id"]) != expected_skills:
    raise ValueError(
        "The question bank must cover KC01 to KC09."
    )

if not set(question_bank["correct_option"]).issubset(
    {"A", "B", "C", "D"}
):
    raise ValueError(
        "Correct options must be A, B, C or D."
    )

for skill_id in sorted(expected_skills):
    skill_questions = question_bank[
        question_bank["skill_id"] == skill_id
    ]

    represented_difficulties = set(
        skill_questions["difficulty"]
    )

    if represented_difficulties != expected_difficulties:
        raise ValueError(
            f"{skill_id} does not contain exactly one "
            "easy, medium and hard question."
        )


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "platform"
    / "sfla_question_bank.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

question_bank.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)

print("=== SFLA Question-Bank Creation ===")
print(f"Questions created: {len(question_bank)}")
print(
    "Knowledge components:",
    question_bank["skill_id"].nunique(),
)
print(
    "\nDifficulty distribution:"
)
print(
    question_bank["difficulty"]
    .value_counts()
    .sort_index()
)
print(f"\nSaved to: {OUTPUT_PATH}")