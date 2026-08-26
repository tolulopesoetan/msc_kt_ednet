import pytest

from adaptive_learning_platform.src.database import (
    initialise_database,
    load_all_interactions,
    load_all_mastery,
    load_interaction_history,
    load_learner_mastery,
    record_interaction,
)

def test_new_learner_receives_skill_specific_initial_mastery(
    tmp_path,
):
    database_path = (
        tmp_path
        / "test_learning.db"
    )

    initialise_database(
        database_path=database_path
    )

    initial_mastery_by_skill = {
        "KC01": 0.15,
        "KC02": 0.35,
    }

    mastery = load_learner_mastery(
        learner_id="new_learner",
        initial_mastery_by_skill=(
            initial_mastery_by_skill
        ),
        database_path=database_path,
    )

    assert mastery["KC01"] == pytest.approx(
        0.15
    )

    assert mastery["KC02"] == pytest.approx(
        0.35
    )


def test_interaction_and_updated_mastery_are_saved(
    tmp_path,
):
    database_path = (
        tmp_path
        / "test_learning.db"
    )

    initialise_database(
        database_path=database_path
    )

    initial_mastery_by_skill = {
        "KC01": 0.20,
        "KC02": 0.30,
    }

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
        model_name=(
            "BKT_SFLA_SIMULATED_PARAMETERS"
        ),
        database_path=database_path,
    )

    mastery = load_learner_mastery(
        learner_id="test_learner",
        initial_mastery_by_skill=(
            initial_mastery_by_skill
        ),
        database_path=database_path,
    )

    assert mastery["KC01"] == pytest.approx(
        0.60
    )

    assert mastery["KC02"] == pytest.approx(
        0.30
    )

    history = load_interaction_history(
        learner_id="test_learner",
        database_path=database_path,
    )

    assert len(history) == 1

    interaction = history.iloc[0]

    assert (
        interaction["session_id"]
        == "test_session"
    )

    assert (
        interaction["interaction_position"]
        == 1
    )

    assert (
        interaction["item_id"]
        == "TEST_001"
    )

    assert (
        interaction["skill_id"]
        == "KC01"
    )

    assert (
        interaction["selected_option"]
        == "C"
    )

    assert (
        interaction["correct_option"]
        == "C"
    )

    assert interaction["actual"] == 1

    assert interaction[
        "response_time_seconds"
    ] == pytest.approx(5.5)

    assert interaction[
        "predicted_probability"
    ] == pytest.approx(0.34)

    assert interaction[
        "mastery_before"
    ] == pytest.approx(0.20)

    assert interaction[
        "mastery_after"
    ] == pytest.approx(0.60)

    assert interaction[
        "model_name"
    ] == "BKT_SFLA_SIMULATED_PARAMETERS"


def test_later_interaction_updates_existing_mastery(
    tmp_path,
):
    database_path = (
        tmp_path
        / "test_learning.db"
    )

    initialise_database(
        database_path=database_path
    )

    question = {
        "item_id": "TEST_002",
        "skill_id": "KC01",
        "correct_option": "A",
    }

    record_interaction(
        learner_id="returning_learner",
        session_id="session_one",
        interaction_position=1,
        question=question,
        selected_option="A",
        correct=True,
        response_time_seconds=4.0,
        predicted_probability=0.40,
        mastery_before=0.20,
        mastery_after=0.55,
        database_path=database_path,
    )

    record_interaction(
        learner_id="returning_learner",
        session_id="session_one",
        interaction_position=2,
        question=question,
        selected_option="B",
        correct=False,
        response_time_seconds=6.0,
        predicted_probability=0.65,
        mastery_before=0.55,
        mastery_after=0.42,
        database_path=database_path,
    )

    mastery = load_learner_mastery(
        learner_id="returning_learner",
        initial_mastery_by_skill={
            "KC01": 0.20,
        },
        database_path=database_path,
    )

    assert mastery["KC01"] == pytest.approx(
        0.42
    )

    history = load_interaction_history(
        learner_id="returning_learner",
        database_path=database_path,
    )

    assert len(history) == 2

    assert history[
        "interaction_position"
    ].tolist() == [1, 2]

    assert history[
        "actual"
    ].tolist() == [1, 0]

    assert history.iloc[-1][
        "mastery_after"
    ] == pytest.approx(0.42)

def test_research_exports_include_all_learners(
    tmp_path,
):
    database_path = (
        tmp_path
        / "test_learning.db"
    )

    initialise_database(
        database_path=database_path
    )

    first_question = {
        "item_id": "TEST_001",
        "skill_id": "KC01",
        "correct_option": "A",
    }

    second_question = {
        "item_id": "TEST_002",
        "skill_id": "KC02",
        "correct_option": "B",
    }

    record_interaction(
        learner_id="learner_one",
        session_id="session_one",
        interaction_position=1,
        question=first_question,
        selected_option="A",
        correct=True,
        response_time_seconds=4.0,
        predicted_probability=0.60,
        mastery_before=0.30,
        mastery_after=0.70,
        database_path=database_path,
    )

    record_interaction(
        learner_id="learner_two",
        session_id="session_two",
        interaction_position=1,
        question=second_question,
        selected_option="A",
        correct=False,
        response_time_seconds=6.0,
        predicted_probability=0.55,
        mastery_before=0.40,
        mastery_after=0.25,
        database_path=database_path,
    )

    interactions = load_all_interactions(
        database_path=database_path
    )

    mastery = load_all_mastery(
        database_path=database_path
    )

    assert len(interactions) == 2
    assert len(mastery) == 2

    assert set(
        interactions["learner_id"]
    ) == {
        "learner_one",
        "learner_two",
    }

    assert set(
        mastery["learner_id"]
    ) == {
        "learner_one",
        "learner_two",
    }

    assert interactions[
        "actual"
    ].tolist() == [1, 0]

    assert set(
        mastery["skill_id"]
    ) == {
        "KC01",
        "KC02",
    }