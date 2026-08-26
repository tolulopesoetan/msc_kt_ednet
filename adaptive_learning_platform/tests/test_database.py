import pytest

from adaptive_learning_platform.src.database import (
    initialise_database,
    load_interaction_history,
    load_learner_mastery,
    record_interaction,
)


def test_new_learner_receives_initial_mastery(
    tmp_path,
):
    database_path = (
        tmp_path / "test_learning.db"
    )

    initialise_database(database_path)

    mastery = load_learner_mastery(
        learner_id="test_learner",
        skill_ids=["KC01", "KC02"],
        initial_mastery=0.20,
        database_path=database_path,
    )

    assert mastery == {
        "KC01": 0.20,
        "KC02": 0.20,
    }


def test_interaction_and_mastery_are_saved(
    tmp_path,
):
    database_path = (
        tmp_path / "test_learning.db"
    )

    initialise_database(database_path)

    question = {
        "item_id": "TEST_001",
        "skill_id": "KC01",
        "correct_option": "C",
    }

    record_interaction(
        learner_id="test_learner",
        session_id="test_session",
        interaction_position=1,
        question=question,
        selected_option="C",
        correct=True,
        response_time_seconds=5.5,
        predicted_probability=0.34,
        mastery_before=0.20,
        mastery_after=0.60,
        database_path=database_path,
    )

    mastery = load_learner_mastery(
        learner_id="test_learner",
        skill_ids=["KC01", "KC02"],
        initial_mastery=0.20,
        database_path=database_path,
    )

    assert mastery["KC01"] == pytest.approx(
        0.60
    )

    assert mastery["KC02"] == pytest.approx(
        0.20
    )

    history = load_interaction_history(
        learner_id="test_learner",
        database_path=database_path,
    )

    assert len(history) == 1

    interaction = history.iloc[0]

    assert interaction["item_id"] == "TEST_001"
    assert interaction["skill_id"] == "KC01"
    assert interaction["actual"] == 1

    assert interaction[
        "predicted_probability"
    ] == pytest.approx(0.34)

    assert interaction[
        "mastery_after"
    ] == pytest.approx(0.60)