from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import time
import uuid

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="SFLA Adaptive Learning Platform",
    page_icon="📘",
    layout="wide",
)


APP_DIRECTORY = Path(__file__).resolve().parent
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


def clamp_probability(value):
    return min(max(float(value), 0.0001), 0.9999)


def probability_correct(prior_mastery):
    prior_mastery = clamp_probability(prior_mastery)

    guess = BKT_PARAMETERS["guess_probability"]
    slip = BKT_PARAMETERS["slip_probability"]

    return (
        prior_mastery * (1 - slip)
        + (1 - prior_mastery) * guess
    )


def update_bkt_mastery(prior_mastery, correct):
    prior_mastery = clamp_probability(prior_mastery)

    learning = BKT_PARAMETERS["learning_probability"]
    guess = BKT_PARAMETERS["guess_probability"]
    slip = BKT_PARAMETERS["slip_probability"]

    if correct:
        numerator = prior_mastery * (1 - slip)
        denominator = numerator + (
            (1 - prior_mastery) * guess
        )
    else:
        numerator = prior_mastery * slip
        denominator = numerator + (
            (1 - prior_mastery) * (1 - guess)
        )

    posterior_mastery = numerator / denominator

    updated_mastery = posterior_mastery + (
        1 - posterior_mastery
    ) * learning

    return clamp_probability(updated_mastery)


def initialise_database():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                interaction_position INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                selected_option TEXT NOT NULL,
                correct_option TEXT NOT NULL,
                actual INTEGER NOT NULL,
                response_time_seconds REAL NOT NULL,
                predicted_probability REAL NOT NULL,
                mastery_before REAL NOT NULL,
                mastery_after REAL NOT NULL,
                model_name TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS learner_mastery (
                learner_id TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                mastery REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (learner_id, skill_id)
            )
            """
        )


@st.cache_data
def load_question_bank():
    if not QUESTION_BANK_PATH.exists():
        raise FileNotFoundError(
            f"Question bank not found: {QUESTION_BANK_PATH}"
        )

    if not ITEM_REGISTER_PATH.exists():
        raise FileNotFoundError(
            f"SFLA item register not found: {ITEM_REGISTER_PATH}"
        )

    question_bank = pd.read_csv(QUESTION_BANK_PATH)

    item_register = pd.read_excel(
        ITEM_REGISTER_PATH,
        sheet_name="sfla_item_skill_map_v1",
    )

    required_columns = {
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

    missing_columns = (
        required_columns - set(question_bank.columns)
    )

    if missing_columns:
        raise ValueError(
            "Question bank is missing columns: "
            f"{sorted(missing_columns)}"
        )

    question_bank["item_id"] = (
        question_bank["item_id"].astype(str)
    )

    question_bank["skill_id"] = (
        question_bank["skill_id"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    question_bank["correct_option"] = (
        question_bank["correct_option"]
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

    if question_bank["item_id"].duplicated().any():
        raise ValueError(
            "Question-bank item IDs must be unique."
        )

    invalid_answers = set(
        question_bank["correct_option"]
    ) - {"A", "B", "C", "D"}

    if invalid_answers:
        raise ValueError(
            "Invalid correct-option values: "
            f"{sorted(invalid_answers)}"
        )

    registered_skills = set(
        item_register["skill_id"]
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
            "Question-bank skills are not present in "
            f"the SFLA register: {sorted(invalid_skills)}"
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

    return question_bank


def load_learner_mastery(learner_id):
    mastery = {
        skill_id: BKT_PARAMETERS["initial_mastery"]
        for skill_id in SKILL_NAMES
    }

    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            """
            SELECT skill_id, mastery
            FROM learner_mastery
            WHERE learner_id = ?
            """,
            (learner_id,),
        ).fetchall()

    for skill_id, mastery_value in rows:
        if skill_id in mastery:
            mastery[skill_id] = float(mastery_value)

    return mastery


def record_interaction(
    learner_id,
    session_id,
    interaction_position,
    question,
    selected_option,
    correct,
    response_time_seconds,
    predicted_probability,
    mastery_before,
    mastery_after,
):
    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO interactions (
                learner_id,
                session_id,
                timestamp,
                interaction_position,
                item_id,
                skill_id,
                selected_option,
                correct_option,
                actual,
                response_time_seconds,
                predicted_probability,
                mastery_before,
                mastery_after,
                model_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                learner_id,
                session_id,
                timestamp,
                interaction_position,
                question["item_id"],
                question["skill_id"],
                selected_option,
                question["correct_option"],
                int(correct),
                response_time_seconds,
                predicted_probability,
                mastery_before,
                mastery_after,
                "BKT",
            ),
        )

        connection.execute(
            """
            INSERT INTO learner_mastery (
                learner_id,
                skill_id,
                mastery,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT (learner_id, skill_id)
            DO UPDATE SET
                mastery = excluded.mastery,
                updated_at = excluded.updated_at
            """,
            (
                learner_id,
                question["skill_id"],
                mastery_after,
                timestamp,
            ),
        )


def load_interaction_history(learner_id):
    with sqlite3.connect(DATABASE_PATH) as connection:
        history = pd.read_sql_query(
            """
            SELECT
                timestamp,
                item_id,
                skill_id,
                actual,
                predicted_probability,
                mastery_before,
                mastery_after,
                response_time_seconds
            FROM interactions
            WHERE learner_id = ?
            ORDER BY id
            """,
            connection,
            params=(learner_id,),
        )

    return history


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
):
    ordered_skills = sorted(
        SKILL_NAMES,
        key=lambda skill_id: mastery[skill_id],
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

        target_difficulty = recommended_difficulty(
            mastery[skill_id]
        )

        difficulty_matches = available_questions[
            available_questions["difficulty"]
            == target_difficulty
        ]

        if not difficulty_matches.empty:
            available_questions = difficulty_matches

        return available_questions.sample(1).iloc[0]

    return None


initialise_database()

try:
    question_bank = load_question_bank()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


st.title("SFLA Adaptive Learning Platform")

st.caption(
    "A knowledge-tracing proof of concept for Sets, "
    "Functions and Linear Algebra."
)


with st.sidebar:
    st.header("Learner session")

    with st.form("learner_form"):
        learner_input = st.text_input(
            "Learner ID",
            placeholder="For example: learner_001",
        )

        start_session = st.form_submit_button(
            "Start session",
            type="primary",
        )

    if start_session:
        learner_id = learner_input.strip()

        if not learner_id:
            st.error("Enter a learner ID.")
        else:
            st.session_state.learner_id = learner_id
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.mastery = (
                load_learner_mastery(learner_id)
            )
            st.session_state.attempted_item_ids = set()
            st.session_state.current_question = None
            st.session_state.feedback = None
            st.session_state.interaction_position = 0
            st.session_state.question_started_at = None
            st.rerun()


if "learner_id" not in st.session_state:
    st.info(
        "Enter a learner ID in the sidebar to begin."
    )
    st.stop()


st.sidebar.success(
    f"Active learner: {st.session_state.learner_id}"
)


learning_tab, progress_tab, about_tab = st.tabs(
    [
        "Learning session",
        "Learner progress",
        "About",
    ]
)


with learning_tab:
    if st.session_state.current_question is None:
        selected_question = select_next_question(
            question_bank=question_bank,
            mastery=st.session_state.mastery,
            attempted_item_ids=(
                st.session_state.attempted_item_ids
            ),
        )

        if selected_question is not None:
            st.session_state.current_question = (
                selected_question.to_dict()
            )
            st.session_state.question_started_at = (
                time.time()
            )

    question = st.session_state.current_question

    if question is None:
        st.success(
            "All available questions have been attempted."
        )

        if st.button("Restart question cycle"):
            st.session_state.attempted_item_ids = set()
            st.session_state.feedback = None
            st.rerun()

    else:
        skill_id = question["skill_id"]
        mastery_before = (
            st.session_state.mastery[skill_id]
        )

        st.subheader(
            f"{skill_id}: {SKILL_NAMES[skill_id]}"
        )

        st.write(
            f"Current estimated mastery: "
            f"**{mastery_before:.1%}**"
        )

        st.markdown(
            f"### {question['question_text']}"
        )

        option_text = {
            "A": question["option_a"],
            "B": question["option_b"],
            "C": question["option_c"],
            "D": question["option_d"],
        }

        if st.session_state.feedback is None:
            with st.form(
                f"question_form_{question['item_id']}"
            ):
                selected_option = st.radio(
                    "Select an answer",
                    options=["A", "B", "C", "D"],
                    index=None,
                    format_func=lambda option: (
                        f"{option}. {option_text[option]}"
                    ),
                    key=(
                        f"answer_{question['item_id']}"
                    ),
                )

                submitted = st.form_submit_button(
                    "Submit answer",
                    type="primary",
                )

            if submitted:
                if selected_option is None:
                    st.warning(
                        "Select an answer before submitting."
                    )
                else:
                    correct = (
                        selected_option
                        == question["correct_option"]
                    )

                    predicted_probability = (
                        probability_correct(
                            mastery_before
                        )
                    )

                    mastery_after = update_bkt_mastery(
                        mastery_before,
                        correct,
                    )

                    response_time = round(
                        time.time()
                        - st.session_state.question_started_at,
                        2,
                    )

                    interaction_position = (
                        st.session_state
                        .interaction_position
                        + 1
                    )

                    record_interaction(
                        learner_id=(
                            st.session_state.learner_id
                        ),
                        session_id=(
                            st.session_state.session_id
                        ),
                        interaction_position=(
                            interaction_position
                        ),
                        question=question,
                        selected_option=selected_option,
                        correct=correct,
                        response_time_seconds=response_time,
                        predicted_probability=(
                            predicted_probability
                        ),
                        mastery_before=mastery_before,
                        mastery_after=mastery_after,
                    )

                    st.session_state.mastery[
                        skill_id
                    ] = mastery_after

                    st.session_state.interaction_position = (
                        interaction_position
                    )

                    st.session_state.feedback = {
                        "correct": correct,
                        "selected_option": selected_option,
                        "mastery_before": mastery_before,
                        "mastery_after": mastery_after,
                        "predicted_probability": (
                            predicted_probability
                        ),
                    }

                    st.rerun()

        else:
            feedback = st.session_state.feedback

            if feedback["correct"]:
                st.success("Correct answer.")
            else:
                st.error("Incorrect answer.")

            correct_option = question["correct_option"]

            st.write(
                "**Correct answer:** "
                f"{correct_option}. "
                f"{option_text[correct_option]}"
            )

            st.info(question["explanation"])

            metric_one, metric_two = st.columns(2)

            metric_one.metric(
                "Predicted correctness",
                (
                    f"{feedback['predicted_probability']:.1%}"
                ),
            )

            metric_two.metric(
                "Updated mastery",
                f"{feedback['mastery_after']:.1%}",
                delta=(
                    feedback["mastery_after"]
                    - feedback["mastery_before"]
                ),
            )

            if st.button(
                "Next question",
                type="primary",
                key=f"next_{question['item_id']}",
            ):
                attempted_items = set(
                    st.session_state.attempted_item_ids
                )

                attempted_items.add(
                    question["item_id"]
                )

                st.session_state.attempted_item_ids = (
                    attempted_items
                )

                answer_widget_key = (
                    f"answer_{question['item_id']}"
                )

                if (
                    answer_widget_key
                    in st.session_state
                ):
                    del st.session_state[
                        answer_widget_key
                    ]

                st.session_state.current_question = None
                st.session_state.feedback = None
                st.session_state.question_started_at = None

                st.rerun()


with progress_tab:
    st.subheader("Knowledge-component mastery")

    mastery_table = pd.DataFrame(
        [
            {
                "skill_id": skill_id,
                "knowledge_component": skill_name,
                "mastery": (
                    st.session_state.mastery[skill_id]
                ),
            }
            for skill_id, skill_name
            in SKILL_NAMES.items()
        ]
    )

    st.bar_chart(
        mastery_table.set_index("skill_id")[
            "mastery"
        ],
        height=350,
    )

    display_mastery = mastery_table.copy()
    display_mastery["mastery"] = (
        display_mastery["mastery"]
        .map(lambda value: f"{value:.1%}")
    )

    st.dataframe(
        display_mastery,
        use_container_width=True,
        hide_index=True,
    )

    history = load_interaction_history(
        st.session_state.learner_id
    )

    if history.empty:
        st.info(
            "Complete a question to generate progress data."
        )
    else:
        total_attempts = len(history)
        overall_accuracy = history["actual"].mean()

        weakest_skill = min(
            st.session_state.mastery,
            key=st.session_state.mastery.get,
        )

        column_one, column_two, column_three = (
            st.columns(3)
        )

        column_one.metric(
            "Questions attempted",
            total_attempts,
        )

        column_two.metric(
            "Overall accuracy",
            f"{overall_accuracy:.1%}",
        )

        column_three.metric(
            "Recommended topic",
            weakest_skill,
        )

        st.write(
            f"Recommended next topic: "
            f"**{SKILL_NAMES[weakest_skill]}**"
        )

        recent_history = history.tail(10).copy()

        recent_history["actual"] = (
            recent_history["actual"].map(
                {1: "Correct", 0: "Incorrect"}
            )
        )

        st.subheader("Recent interactions")

        st.dataframe(
            recent_history,
            use_container_width=True,
            hide_index=True,
        )


with about_tab:
    st.subheader("About this prototype")

    st.write(
        """
        This platform demonstrates how knowledge tracing can
        support adaptive question sequencing in the SFLA
        domain. Bayesian Knowledge Tracing updates a separate
        mastery estimate for each knowledge component after
        every response.

        Questions are selected from the learner's weakest
        available knowledge component. Difficulty is chosen
        from the learner's estimated mastery level.

        The current BKT parameters are demonstration values.
        They should later be replaced with parameters estimated
        from appropriate learner-interaction data. The platform
        is therefore a functional proof of concept and not
        evidence of improved learning outcomes.
        """
    )