from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from adaptive_learning_platform.src.bkt import (
    probability_correct,
    update_mastery,
)
from adaptive_learning_platform.src.config import (
    PROJECT_ROOT,
    SKILL_NAMES,
)
from adaptive_learning_platform.src.data_loader import (
    load_question_bank,
)
from adaptive_learning_platform.src.database import (
    initialise_database,
    load_all_interactions,
    load_all_mastery,
    load_interaction_history,
    load_learner_mastery,
    record_interaction,
)
from adaptive_learning_platform.src.parameter_loader import (
    load_bkt_parameter_artifact,
)
from adaptive_learning_platform.src.recommender import (
    recommended_difficulty,
    select_next_question,
)


evaluation_records = []


def add_result(
    check,
    status,
    result,
    requirement,
):
    evaluation_records.append(
        {
            "check": check,
            "status": status,
            "result": result,
            "requirement": requirement,
        }
    )


question_bank = load_question_bank()

parameter_artifact = (
    load_bkt_parameter_artifact()
)

parameters_by_skill = (
    parameter_artifact["skills"]
)


question_bank_valid = (
    len(question_bank) >= 27
    and question_bank[
        "skill_id"
    ].nunique() == len(SKILL_NAMES)
)

add_result(
    check="Question-bank readiness",
    status=(
        "PASSED"
        if question_bank_valid
        else "FAILED"
    ),
    result=(
        f"{len(question_bank)} questions; "
        f"{question_bank['skill_id'].nunique()} "
        "skills"
    ),
    requirement=(
        "At least 27 questions covering "
        "all nine skills"
    ),
)


parameter_coverage_valid = (
    set(parameters_by_skill)
    == set(SKILL_NAMES)
)

add_result(
    check="BKT parameter coverage",
    status=(
        "PASSED"
        if parameter_coverage_valid
        else "FAILED"
    ),
    result=(
        f"{len(parameters_by_skill)} "
        "skill parameter sets"
    ),
    requirement=(
        "One validated parameter set for "
        "each SFLA skill"
    ),
)


add_result(
    check="Parameter provenance",
    status="PASSED",
    result=parameter_artifact[
        "data_status"
    ],
    requirement=(
        "Data status must be explicitly "
        "recorded"
    ),
)


initial_mastery_by_skill = {
    skill_id: parameters[
        "initial_mastery"
    ]
    for skill_id, parameters
    in parameters_by_skill.items()
}


predictions_in_range = []
updates_in_range = []
response_ordering_checks = []


for skill_id, parameters in (
    parameters_by_skill.items()
):
    prior_mastery = parameters[
        "initial_mastery"
    ]

    predicted_probability = (
        probability_correct(
            prior_mastery=prior_mastery,
            parameters=parameters,
        )
    )

    mastery_after_correct = (
        update_mastery(
            prior_mastery=prior_mastery,
            correct=True,
            parameters=parameters,
        )
    )

    mastery_after_incorrect = (
        update_mastery(
            prior_mastery=prior_mastery,
            correct=False,
            parameters=parameters,
        )
    )

    predictions_in_range.append(
        0 < predicted_probability < 1
    )

    updates_in_range.extend(
        [
            0 < mastery_after_correct < 1,
            0 < mastery_after_incorrect < 1,
        ]
    )

    response_ordering_checks.append(
        mastery_after_correct
        >= mastery_after_incorrect
    )


add_result(
    check="BKT probability range",
    status=(
        "PASSED"
        if all(predictions_in_range)
        else "FAILED"
    ),
    result=(
        f"{sum(predictions_in_range)}/"
        f"{len(predictions_in_range)} valid"
    ),
    requirement=(
        "Every predicted probability must "
        "lie between zero and one"
    ),
)


add_result(
    check="BKT mastery-update range",
    status=(
        "PASSED"
        if all(updates_in_range)
        else "FAILED"
    ),
    result=(
        f"{sum(updates_in_range)}/"
        f"{len(updates_in_range)} valid"
    ),
    requirement=(
        "Every updated mastery value must "
        "lie between zero and one"
    ),
)


ordering_status = (
    "PASSED"
    if all(response_ordering_checks)
    else "WARNING"
)

add_result(
    check=(
        "Correct-versus-incorrect "
        "update ordering"
    ),
    status=ordering_status,
    result=(
        f"{sum(response_ordering_checks)}/"
        f"{len(response_ordering_checks)} "
        "skills ordered as expected"
    ),
    requirement=(
        "A correct response should not "
        "produce less mastery than an "
        "incorrect response"
    ),
)


weakest_skill = min(
    initial_mastery_by_skill,
    key=lambda skill_id: (
        initial_mastery_by_skill[
            skill_id
        ],
        skill_id,
    ),
)


selected_question = (
    select_next_question(
        question_bank=question_bank,
        mastery=initial_mastery_by_skill,
        attempted_item_ids=set(),
        random_state=42,
    )
)


if selected_question is None:
    add_result(
        check="Adaptive question selection",
        status="FAILED",
        result="No question selected",
        requirement=(
            "The recommender must return "
            "an available question"
        ),
    )

    raise RuntimeError(
        "The recommender returned no question."
    )


selected_skill = selected_question[
    "skill_id"
]

selected_difficulty = selected_question[
    "difficulty"
]

expected_difficulty = (
    recommended_difficulty(
        initial_mastery_by_skill[
            selected_skill
        ]
    )
)


add_result(
    check="Weakest-skill prioritisation",
    status=(
        "PASSED"
        if selected_skill == weakest_skill
        else "FAILED"
    ),
    result=(
        f"Selected {selected_skill}; "
        f"weakest was {weakest_skill}"
    ),
    requirement=(
        "The first question must target "
        "the lowest-mastery skill"
    ),
)


add_result(
    check="Difficulty adaptation",
    status=(
        "PASSED"
        if selected_difficulty
        == expected_difficulty
        else "FAILED"
    ),
    result=(
        f"Selected {selected_difficulty}; "
        f"expected {expected_difficulty}"
    ),
    requirement=(
        "Question difficulty must match "
        "the mastery-threshold rule"
    ),
)


selected_parameters = (
    parameters_by_skill[selected_skill]
)

selected_prior = (
    initial_mastery_by_skill[
        selected_skill
    ]
)

selected_prediction = (
    probability_correct(
        prior_mastery=selected_prior,
        parameters=selected_parameters,
    )
)

selected_updated_mastery = (
    update_mastery(
        prior_mastery=selected_prior,
        correct=True,
        parameters=selected_parameters,
    )
)


with TemporaryDirectory() as (
    temporary_directory
):
    test_database_path = (
        Path(temporary_directory)
        / "functional_test.db"
    )

    initialise_database(
        database_path=test_database_path
    )

    loaded_initial_mastery = (
        load_learner_mastery(
            learner_id=(
                "FUNCTIONAL_TEST_LEARNER"
            ),
            initial_mastery_by_skill=(
                initial_mastery_by_skill
            ),
            database_path=(
                test_database_path
            ),
        )
    )

    record_interaction(
        learner_id=(
            "FUNCTIONAL_TEST_LEARNER"
        ),
        session_id=(
            "FUNCTIONAL_TEST_SESSION"
        ),
        interaction_position=1,
        question=(
            selected_question.to_dict()
        ),
        selected_option=(
            selected_question[
                "correct_option"
            ]
        ),
        correct=True,
        response_time_seconds=1.0,
        predicted_probability=(
            selected_prediction
        ),
        mastery_before=selected_prior,
        mastery_after=(
            selected_updated_mastery
        ),
        model_name=(
            "BKT_SFLA_FUNCTIONAL_TEST"
        ),
        database_path=test_database_path,
    )

    loaded_updated_mastery = (
        load_learner_mastery(
            learner_id=(
                "FUNCTIONAL_TEST_LEARNER"
            ),
            initial_mastery_by_skill=(
                initial_mastery_by_skill
            ),
            database_path=(
                test_database_path
            ),
        )
    )

    learner_history = (
        load_interaction_history(
            learner_id=(
                "FUNCTIONAL_TEST_LEARNER"
            ),
            database_path=(
                test_database_path
            ),
        )
    )

    exported_interactions = (
        load_all_interactions(
            database_path=(
                test_database_path
            )
        )
    )

    exported_mastery = (
        load_all_mastery(
            database_path=(
                test_database_path
            )
        )
    )

    initial_load_valid = all(
        abs(
            loaded_initial_mastery[
                skill_id
            ]
            - initial_mastery_by_skill[
                skill_id
            ]
        )
        < 1e-12
        for skill_id in SKILL_NAMES
    )

    persistence_valid = (
        len(learner_history) == 1
        and abs(
            loaded_updated_mastery[
                selected_skill
            ]
            - selected_updated_mastery
        )
        < 1e-12
    )

    exports_valid = (
        len(exported_interactions) == 1
        and len(exported_mastery) == 1
    )


add_result(
    check=(
        "Skill-specific initial "
        "mastery loading"
    ),
    status=(
        "PASSED"
        if initial_load_valid
        else "FAILED"
    ),
    result=(
        f"{len(loaded_initial_mastery)} "
        "skills initialised"
    ),
    requirement=(
        "A new learner must receive all "
        "nine fitted prior values"
    ),
)


add_result(
    check=(
        "Interaction and mastery "
        "persistence"
    ),
    status=(
        "PASSED"
        if persistence_valid
        else "FAILED"
    ),
    result=(
        f"{len(learner_history)} "
        "interaction persisted"
    ),
    requirement=(
        "The response and updated mastery "
        "must survive a database reload"
    ),
)


add_result(
    check="Research export queries",
    status=(
        "PASSED"
        if exports_valid
        else "FAILED"
    ),
    result=(
        f"{len(exported_interactions)} "
        "interaction rows; "
        f"{len(exported_mastery)} "
        "mastery rows"
    ),
    requirement=(
        "Interaction and mastery exports "
        "must return stored records"
    ),
)


evaluation_table = pd.DataFrame(
    evaluation_records
)

results_directory = (
    PROJECT_ROOT
    / "results"
    / "adaptive_platform"
)

results_directory.mkdir(
    parents=True,
    exist_ok=True,
)

scorecard_path = (
    results_directory
    / "platform_functional_scorecard.csv"
)

summary_path = (
    results_directory
    / "platform_functional_summary.json"
)

evaluation_table.to_csv(
    scorecard_path,
    index=False,
)


status_counts = (
    evaluation_table[
        "status"
    ].value_counts().to_dict()
)


summary = {
    "evaluated_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "domain": "sfla",
    "parameter_data_status": (
        parameter_artifact[
            "data_status"
        ]
    ),
    "checks": int(
        len(evaluation_table)
    ),
    "passed": int(
        status_counts.get("PASSED", 0)
    ),
    "warnings": int(
        status_counts.get("WARNING", 0)
    ),
    "failed": int(
        status_counts.get("FAILED", 0)
    ),
    "scorecard": str(
        scorecard_path.relative_to(
            PROJECT_ROOT
        )
    ),
}


with summary_path.open(
    "w",
    encoding="utf-8",
) as summary_file:
    json.dump(
        summary,
        summary_file,
        indent=2,
    )


print(
    "=== SFLA Platform "
    "Functional Evaluation ==="
)

print(
    evaluation_table.to_string(
        index=False
    )
)

print()
print("Status counts:")

print(
    evaluation_table[
        "status"
    ].value_counts().to_string()
)

print()
print(
    "Scorecard saved to:",
    scorecard_path,
)

print(
    "Summary saved to:",
    summary_path,
)


if summary["failed"]:
    raise SystemExit(1)