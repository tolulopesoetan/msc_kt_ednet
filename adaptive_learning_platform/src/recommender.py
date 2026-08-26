def recommended_difficulty(mastery):
    if mastery < 0.40:
        return "easy"

    if mastery < 0.70:
        return "medium"

    return "hard"


def select_next_question(
    question_bank,
    mastery,
    attempted_item_ids,
    random_state=None,
):
    ordered_skills = sorted(
        mastery,
        key=lambda skill_id: (
            mastery[skill_id],
            skill_id,
        ),
    )

    for skill_id in ordered_skills:
        available_questions = question_bank[
            (question_bank["skill_id"] == skill_id)
            & (
                ~question_bank["item_id"].isin(
                    attempted_item_ids
                )
            )
        ]

        if available_questions.empty:
            continue

        target_difficulty = (
            recommended_difficulty(
                mastery[skill_id]
            )
        )

        difficulty_matches = (
            available_questions[
                available_questions["difficulty"]
                == target_difficulty
            ]
        )

        if not difficulty_matches.empty:
            available_questions = (
                difficulty_matches
            )

        return available_questions.sample(
            1,
            random_state=random_state,
        ).iloc[0]

    return None