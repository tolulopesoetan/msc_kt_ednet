from datetime import datetime, timezone
import hmac
import json
import time
import uuid

import pandas as pd
import streamlit as st

from src.bkt import (
    probability_correct,
    update_mastery,
)
from src.config import SKILL_NAMES
from src.data_loader import load_question_bank
from src.database import (
    initialise_database,
    load_all_interactions,
    load_all_mastery,
    load_interaction_history,
    load_learner_interactions,
    load_learner_mastery,
    load_learner_mastery_table,
    load_session_history,
    record_interaction,
)
from src.parameter_loader import (
    load_bkt_parameter_artifact,
)
from src.recommender import (
    select_next_question,
)
from src.research_models import (
    load_research_models_safely,
    predict_research_probabilities,
)


st.set_page_config(
    page_title="SFLA Adaptive Learning Platform",
    page_icon="📘",
    layout="wide",
)


@st.cache_data
def get_platform_inputs():
    return (
        load_question_bank(),
        load_bkt_parameter_artifact(),
    )


@st.cache_resource(show_spinner=False)
def get_research_model_bundle():
    return load_research_models_safely()


def generate_research_prediction(
    enabled,
    research_bundle,
    research_error,
    learner_id,
    next_skill_id,
):
    """Generate optional predictions from prior responses only."""

    if not enabled:
        return {
            "available": False,
            "status": "disabled",
            "reason": (
                "Enable optional research predictions "
                "in the sidebar to use DKT and SAKT."
            ),
            "dkt_probability": None,
            "sakt_probability": None,
            "data_status": None,
        }

    if research_bundle is None:
        return {
            "available": False,
            "status": "model_unavailable",
            "reason": (
                research_error
                or "The optional research models are unavailable."
            ),
            "dkt_probability": None,
            "sakt_probability": None,
            "data_status": None,
        }

    completed_history = load_interaction_history(
        learner_id
    )

    try:
        prediction = predict_research_probabilities(
            bundle=research_bundle,
            completed_history=completed_history[
                ["skill_id", "actual"]
            ],
            next_skill_id=next_skill_id,
        )
    except Exception as error:
        return {
            "available": False,
            "status": "prediction_error",
            "reason": str(error),
            "dkt_probability": None,
            "sakt_probability": None,
            "data_status": (
                research_bundle.metadata.get(
                    "data_status"
                )
            ),
        }

    prediction["status"] = (
        "available"
        if prediction["available"]
        else "insufficient_history"
    )

    return prediction


def start_learner_session(
    learner_id,
    initial_mastery,
):
    """Initialise a clean browser session for one learner."""

    for state_key in list(st.session_state):
        if state_key.startswith("answer_"):
            del st.session_state[state_key]

    st.session_state.learner_id = learner_id
    st.session_state.session_id = str(
        uuid.uuid4()
    )
    st.session_state.session_started_at = (
        datetime.now(timezone.utc).isoformat()
    )
    st.session_state.session_ended = False
    st.session_state.admin_export_unlocked = False
    st.session_state.mastery = (
        load_learner_mastery(
            learner_id=learner_id,
            initial_mastery_by_skill=(
                initial_mastery
            ),
        )
    )
    st.session_state.attempted_item_ids = set()
    st.session_state.current_question = None
    st.session_state.feedback = None
    st.session_state.interaction_position = 0
    st.session_state.question_started_at = None
    st.session_state.current_research_prediction = None
    st.session_state.research_prediction_key = None


def build_mastery_trajectory(
    history,
    initial_mastery,
):
    """Reconstruct every skill's mastery after each attempt."""

    mastery_state = {
        skill_id: float(value)
        for skill_id, value
        in initial_mastery.items()
    }

    trajectory_rows = [
        {
            "attempt": 0,
            **mastery_state,
        }
    ]

    for attempt_number, interaction in enumerate(
        history.itertuples(index=False),
        start=1,
    ):
        if interaction.skill_id in mastery_state:
            mastery_state[interaction.skill_id] = float(
                interaction.mastery_after
            )

        trajectory_rows.append(
            {
                "attempt": attempt_number,
                **mastery_state,
            }
        )

    return pd.DataFrame(trajectory_rows)


def build_common_prediction_table(interactions):
    """Create a like-for-like table and quantify model spread."""

    output_columns = [
        "learner_id",
        "session_id",
        "interaction_position",
        "skill_id",
        "actual",
        "bkt_probability",
        "dkt_probability",
        "sakt_probability",
        "bkt_dkt_gap",
        "bkt_sakt_gap",
        "dkt_sakt_gap",
        "probability_spread",
        "highest_model",
        "lowest_model",
        "disagreement_level",
    ]

    required_columns = {
        "learner_id",
        "session_id",
        "interaction_position",
        "skill_id",
        "actual",
        "predicted_probability",
        "dkt_probability",
        "sakt_probability",
    }

    if (
        interactions.empty
        or not required_columns.issubset(
            interactions.columns
        )
    ):
        return pd.DataFrame(
            columns=output_columns
        )

    common = (
        interactions.dropna(
            subset=[
                "predicted_probability",
                "dkt_probability",
                "sakt_probability",
            ]
        )[
            [
                "learner_id",
                "session_id",
                "interaction_position",
                "skill_id",
                "actual",
                "predicted_probability",
                "dkt_probability",
                "sakt_probability",
            ]
        ]
        .rename(
            columns={
                "predicted_probability": (
                    "bkt_probability"
                )
            }
        )
        .reset_index(drop=True)
    )

    if common.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    probability_columns = [
        "bkt_probability",
        "dkt_probability",
        "sakt_probability",
    ]

    common["bkt_dkt_gap"] = (
        common["bkt_probability"]
        - common["dkt_probability"]
    ).abs()
    common["bkt_sakt_gap"] = (
        common["bkt_probability"]
        - common["sakt_probability"]
    ).abs()
    common["dkt_sakt_gap"] = (
        common["dkt_probability"]
        - common["sakt_probability"]
    ).abs()
    common["probability_spread"] = (
        common[probability_columns].max(axis=1)
        - common[probability_columns].min(axis=1)
    )

    model_labels = {
        "bkt_probability": "BKT",
        "dkt_probability": "DKT",
        "sakt_probability": "SAKT",
    }

    common["highest_model"] = (
        common[probability_columns]
        .idxmax(axis=1)
        .map(model_labels)
    )
    common["lowest_model"] = (
        common[probability_columns]
        .idxmin(axis=1)
        .map(model_labels)
    )

    common["disagreement_level"] = (
        common["probability_spread"].map(
            lambda value: (
                "High"
                if value >= 0.20
                else (
                    "Moderate"
                    if value >= 0.10
                    else "Low"
                )
            )
        )
    )

    return common[output_columns]


def get_admin_password():
    """Read the administrator export password without exposing it."""

    try:
        password = st.secrets.get(
            "SFLA_ADMIN_PASSWORD",
            "",
        )
    except Exception:
        return ""

    return (
        password.strip()
        if isinstance(password, str)
        else ""
    )


def render_session_summary(
    learner_id,
    session_id,
    mastery,
):
    """Display and export a summary of the completed session."""

    st.subheader("Session summary")

    session_history = load_session_history(
        learner_id=learner_id,
        session_id=session_id,
    )

    if session_history.empty:
        st.info(
            "This session ended before any questions "
            "were completed. Start a new session from "
            "the sidebar when you are ready."
        )
        return

    attempts = len(session_history)
    accuracy = session_history["actual"].mean()
    strongest_skill = max(
        mastery,
        key=mastery.get,
    )
    weakest_skill = min(
        mastery,
        key=mastery.get,
    )

    (
        summary_column_one,
        summary_column_two,
        summary_column_three,
        summary_column_four,
    ) = st.columns(4)

    summary_column_one.metric(
        "Questions completed",
        attempts,
    )
    summary_column_two.metric(
        "Session accuracy",
        f"{accuracy:.1%}",
    )
    summary_column_three.metric(
        "Strongest skill",
        (
            f"{strongest_skill} "
            f"({mastery[strongest_skill]:.1%})"
        ),
    )
    summary_column_four.metric(
        "Recommended next skill",
        (
            f"{weakest_skill} "
            f"({mastery[weakest_skill]:.1%})"
        ),
    )

    st.write(
        "Strongest topic: "
        f"**{SKILL_NAMES[strongest_skill]}**"
    )
    st.write(
        "Recommended next topic: "
        f"**{SKILL_NAMES[weakest_skill]}**"
    )

    summary_columns = [
        "interaction_position",
        "item_id",
        "skill_id",
        "actual",
        "response_time_seconds",
        "predicted_probability",
        "dkt_probability",
        "sakt_probability",
        "mastery_before",
        "mastery_after",
    ]

    st.dataframe(
        session_history[summary_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download session summary",
        data=(
            session_history.to_csv(index=False)
            .encode("utf-8")
        ),
        file_name=(
            "sfla_session_"
            f"{session_id[:8]}.csv"
        ),
        mime="text/csv",
        key=f"session_summary_{session_id}",
    )

    st.info(
        "This session is closed. Start a new manual "
        "or anonymous session from the sidebar."
    )


def render_research_export_scope(
    interactions,
    mastery,
    scope_label,
    file_prefix,
    parameter_status,
    question_bank_size,
    validity_interpretation,
):
    """Render metrics, disagreement analysis and downloads."""

    st.markdown(f"### {scope_label}")

    if interactions.empty:
        st.info(
            "No interactions are available for this "
            "export scope."
        )
        return

    total_interactions = len(interactions)
    total_learners = interactions[
        "learner_id"
    ].nunique()
    represented_skills = interactions[
        "skill_id"
    ].nunique()
    observed_accuracy = interactions[
        "actual"
    ].mean()

    (
        export_column_one,
        export_column_two,
        export_column_three,
        export_column_four,
    ) = st.columns(4)

    export_column_one.metric(
        "Recorded interactions",
        total_interactions,
    )
    export_column_two.metric(
        "Learners",
        total_learners,
    )
    export_column_three.metric(
        "Skills represented",
        f"{represented_skills}/{len(SKILL_NAMES)}",
    )
    export_column_four.metric(
        "Observed accuracy",
        f"{observed_accuracy:.1%}",
    )

    st.subheader("Interaction preview")

    interaction_preview = interactions.tail(20).copy()
    interaction_preview["actual"] = (
        interaction_preview["actual"].map(
            {
                1: "Correct",
                0: "Incorrect",
            }
        )
    )

    st.dataframe(
        interaction_preview,
        use_container_width=True,
        hide_index=True,
    )

    common_predictions = (
        build_common_prediction_table(
            interactions
        )
    )

    st.subheader("Model disagreement analysis")

    if common_predictions.empty:
        st.info(
            "No interaction currently has predictions "
            "from all three models. Enable DKT and SAKT, "
            "then complete at least two questions."
        )
    else:
        high_disagreement_count = int(
            (
                common_predictions[
                    "disagreement_level"
                ]
                == "High"
            ).sum()
        )
        mean_spread = common_predictions[
            "probability_spread"
        ].mean()

        (
            disagreement_column_one,
            disagreement_column_two,
            disagreement_column_three,
        ) = st.columns(3)

        disagreement_column_one.metric(
            "Common prediction rows",
            len(common_predictions),
        )
        disagreement_column_two.metric(
            "Mean probability spread",
            f"{mean_spread:.1%}",
        )
        disagreement_column_three.metric(
            "High-disagreement rows",
            high_disagreement_count,
        )

        st.caption(
            "Disagreement is the difference between the "
            "highest and lowest model probabilities. "
            "Low is below 10 percentage points, moderate "
            "is 10–19.9 points and high is at least 20 points."
        )

        st.dataframe(
            common_predictions.tail(20),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Current mastery states")

    mastery_preview = mastery.copy()

    if not mastery_preview.empty:
        mastery_preview["mastery"] = (
            mastery_preview["mastery"].map(
                lambda value: f"{value:.1%}"
            )
        )

    st.dataframe(
        mastery_preview,
        use_container_width=True,
        hide_index=True,
    )

    export_timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    export_manifest = {
        "exported_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "export_scope": scope_label,
        "domain": "sfla",
        "adaptive_engine": "BKT",
        "optional_research_models": [
            "DKT",
            "Causal SAKT",
        ],
        "research_models_control_recommendations": False,
        "parameter_data_status": parameter_status,
        "interaction_rows": int(total_interactions),
        "unique_learners": int(total_learners),
        "represented_skills": int(
            represented_skills
        ),
        "question_bank_size": int(
            question_bank_size
        ),
        "common_prediction_rows": int(
            len(common_predictions)
        ),
        "high_disagreement_rows": int(
            (
                common_predictions[
                    "disagreement_level"
                ]
                == "High"
            ).sum()
            if not common_predictions.empty
            else 0
        ),
        "validity_interpretation": (
            validity_interpretation
        ),
        "privacy_note": (
            "Learner IDs must be pseudonymous. "
            "Application-wide exports require "
            "administrator authentication."
        ),
    }

    interaction_csv = interactions.to_csv(
        index=False
    ).encode("utf-8")
    mastery_csv = mastery.to_csv(
        index=False
    ).encode("utf-8")
    disagreement_csv = (
        common_predictions.to_csv(index=False)
        .encode("utf-8")
    )
    manifest_json = json.dumps(
        export_manifest,
        indent=2,
    ).encode("utf-8")

    st.subheader("Download research artifacts")

    (
        download_column_one,
        download_column_two,
        download_column_three,
        download_column_four,
    ) = st.columns(4)

    download_column_one.download_button(
        label="Download interactions",
        data=interaction_csv,
        file_name=(
            f"{file_prefix}_interactions_"
            f"{export_timestamp}.csv"
        ),
        mime="text/csv",
        key=f"{file_prefix}_interactions_download",
    )
    download_column_two.download_button(
        label="Download mastery states",
        data=mastery_csv,
        file_name=(
            f"{file_prefix}_mastery_"
            f"{export_timestamp}.csv"
        ),
        mime="text/csv",
        key=f"{file_prefix}_mastery_download",
    )
    download_column_three.download_button(
        label="Download export manifest",
        data=manifest_json,
        file_name=(
            f"{file_prefix}_manifest_"
            f"{export_timestamp}.json"
        ),
        mime="application/json",
        key=f"{file_prefix}_manifest_download",
    )
    download_column_four.download_button(
        label="Download disagreement analysis",
        data=disagreement_csv,
        file_name=(
            f"{file_prefix}_disagreement_"
            f"{export_timestamp}.csv"
        ),
        mime="text/csv",
        disabled=common_predictions.empty,
        key=f"{file_prefix}_disagreement_download",
    )


initialise_database()

try:
    (
        question_bank,
        bkt_artifact,
    ) = get_platform_inputs()

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


bkt_parameters_by_skill = (
    bkt_artifact["skills"]
)

parameter_data_status = (
    bkt_artifact["data_status"]
)

initial_mastery_by_skill = {
    skill_id: parameters[
        "initial_mastery"
    ]
    for skill_id, parameters
    in bkt_parameters_by_skill.items()
}


st.title("SFLA Adaptive Learning Platform")

st.caption(
    "A knowledge-tracing proof of concept for "
    "Sets, Functions and Linear Algebra."
)


with st.sidebar:
    st.header("Learner session")

    with st.form("learner_form"):
        learner_input = st.text_input(
            "Pseudonymous learner ID",
            placeholder="For example: learner_001",
            help=(
                "Use a study code or invented identifier. "
                "Do not enter a name, email address or "
                "student number."
            ),
        )

        start_session = (
            st.form_submit_button(
                "Start session",
                type="primary",
            )
        )

    start_anonymous_demo = st.button(
        "Start anonymous demo",
        use_container_width=True,
        help=(
            "Creates a random pseudonymous learner ID "
            "for a demonstration session."
        ),
    )

    if start_session:
        learner_id = learner_input.strip()

        if not learner_id:
            st.error("Enter a learner ID.")
        else:
            start_learner_session(
                learner_id=learner_id,
                initial_mastery=(
                    initial_mastery_by_skill
                ),
            )
            st.rerun()

    if start_anonymous_demo:
        anonymous_learner_id = (
            "demo_"
            f"{uuid.uuid4().hex[:12]}"
        )

        start_learner_session(
            learner_id=anonymous_learner_id,
            initial_mastery=(
                initial_mastery_by_skill
            ),
        )
        st.rerun()

    st.caption(
        "Only pseudonymous IDs should be used. "
        "The hosted demonstration is not intended "
        "for personal or sensitive learner data."
    )

    st.divider()
    st.subheader("Optional research predictions")

    enable_research_predictions = st.toggle(
        "Enable DKT and SAKT",
        value=False,
        help=(
            "Displays research-only predictions after an "
            "answer is submitted. BKT remains the adaptive "
            "engine."
        ),
    )

    if enable_research_predictions:
        with st.spinner(
            "Loading research models..."
        ):
            (
                research_model_bundle,
                research_model_error,
            ) = get_research_model_bundle()

        if research_model_bundle is None:
            st.warning(
                "Research predictions unavailable. "
                "BKT remains operational."
            )
        else:
            st.success(
                "DKT and SAKT loaded for "
                "research display only."
            )
    else:
        research_model_bundle = None
        research_model_error = None


if "learner_id" not in st.session_state:
    st.info(
        "Enter a learner ID in the sidebar "
        "to begin."
    )
    st.stop()


active_learner = (
    st.session_state.learner_id
)

st.sidebar.success(
    f"Active learner: {active_learner}"
)

if not st.session_state.get(
    "session_ended",
    False,
):
    if st.sidebar.button(
        "End current session",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.session_ended = True
        st.session_state.current_question = None
        st.session_state.feedback = None
        st.session_state.question_started_at = None
        st.session_state.current_research_prediction = None
        st.session_state.research_prediction_key = None
        st.rerun()
else:
    st.sidebar.info(
        "This session has ended. Start another "
        "session above to continue."
    )

st.sidebar.caption(
    "BKT parameter data: "
    f"{parameter_data_status}"
)


(
    learning_tab,
    progress_tab,
    research_tab,
    about_tab,
) = st.tabs(
    [
        "Learning session",
        "Learner progress",
        "Research export",
        "About",
    ]
)


with learning_tab:
    if parameter_data_status == "SIMULATED":
        st.warning(
            "This prototype uses BKT parameters "
            "fitted to simulated SFLA interactions. "
            "Its recommendations demonstrate system "
            "operation and are not validated learning "
            "recommendations."
        )

    session_ended = st.session_state.get(
        "session_ended",
        False,
    )

    if (
        not session_ended
        and st.session_state.current_question is None
    ):
        selected_question = (
            select_next_question(
                question_bank=question_bank,
                mastery=(
                    st.session_state.mastery
                ),
                attempted_item_ids=(
                    st.session_state
                    .attempted_item_ids
                ),
            )
        )

        if selected_question is not None:
            st.session_state.current_question = (
                selected_question.to_dict()
            )

            st.session_state.question_started_at = (
                time.time()
            )

            st.session_state.current_research_prediction = (
                None
            )

            st.session_state.research_prediction_key = (
                None
            )

    question = (
        None
        if session_ended
        else st.session_state.current_question
    )

    if question is None and session_ended:
        render_session_summary(
            learner_id=active_learner,
            session_id=(
                st.session_state.session_id
            ),
            mastery=st.session_state.mastery,
        )

    elif question is None:
        st.success(
            "All available questions have "
            "been attempted."
        )

        if st.button("Restart question cycle"):
            st.session_state.attempted_item_ids = (
                set()
            )

            st.session_state.feedback = None
            st.session_state.current_question = None

            st.session_state.question_started_at = (
                None
            )

            st.session_state.current_research_prediction = (
                None
            )

            st.session_state.research_prediction_key = (
                None
            )

            st.rerun()

    else:
        skill_id = question["skill_id"]
        skill_name = SKILL_NAMES[skill_id]

        skill_parameters = (
            bkt_parameters_by_skill[skill_id]
        )

        mastery_before = (
            st.session_state.mastery[skill_id]
        )

        research_prediction_key = (
            st.session_state.session_id,
            question["item_id"],
            st.session_state.interaction_position,
            enable_research_predictions,
        )

        if (
            st.session_state.feedback is None
            and st.session_state.get(
                "research_prediction_key"
            ) != research_prediction_key
        ):
            st.session_state.current_research_prediction = (
                generate_research_prediction(
                    enabled=enable_research_predictions,
                    research_bundle=(
                        research_model_bundle
                    ),
                    research_error=(
                        research_model_error
                    ),
                    learner_id=active_learner,
                    next_skill_id=skill_id,
                )
            )

            st.session_state.research_prediction_key = (
                research_prediction_key
            )

        st.subheader(
            f"{skill_id}: {skill_name}"
        )

        st.write(
            "Current estimated mastery: "
            f"**{mastery_before:.1%}**"
        )

        st.caption(
            "Selected difficulty: "
            f"{question['difficulty'].title()}"
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
            form_key = (
                "question_form_"
                f"{question['item_id']}"
            )

            answer_key = (
                "answer_"
                f"{question['item_id']}"
            )

            with st.form(form_key):
                selected_option = st.radio(
                    "Select an answer",
                    options=[
                        "A",
                        "B",
                        "C",
                        "D",
                    ],
                    index=None,
                    format_func=lambda option: (
                        f"{option}. "
                        f"{option_text[option]}"
                    ),
                    key=answer_key,
                )

                submitted = (
                    st.form_submit_button(
                        "Submit answer",
                        type="primary",
                    )
                )

            if submitted:
                if selected_option is None:
                    st.warning(
                        "Select an answer before "
                        "submitting."
                    )
                else:
                    correct_option = question[
                        "correct_option"
                    ]

                    correct = (
                        selected_option
                        == correct_option
                    )

                    predicted_probability = (
                        probability_correct(
                            prior_mastery=(
                                mastery_before
                            ),
                            parameters=(
                                skill_parameters
                            ),
                        )
                    )

                    mastery_after = (
                        update_mastery(
                            prior_mastery=(
                                mastery_before
                            ),
                            correct=correct,
                            parameters=(
                                skill_parameters
                            ),
                        )
                    )

                    response_time = round(
                        time.time()
                        - st.session_state
                        .question_started_at,
                        2,
                    )

                    interaction_position = (
                        st.session_state
                        .interaction_position
                        + 1
                    )

                    interaction_model_name = (
                        "BKT_SFLA_"
                        f"{parameter_data_status}_"
                        "PARAMETERS"
                    )

                    research_prediction = (
                        st.session_state.get(
                            "current_research_prediction"
                        )
                        or {
                            "available": False,
                            "status": "not_recorded",
                            "reason": (
                                "No optional research prediction "
                                "was generated."
                            ),
                            "dkt_probability": None,
                            "sakt_probability": None,
                            "data_status": None,
                        }
                    )

                    record_interaction(
                        learner_id=(
                            st.session_state
                            .learner_id
                        ),
                        session_id=(
                            st.session_state
                            .session_id
                        ),
                        interaction_position=(
                            interaction_position
                        ),
                        question=question,
                        selected_option=(
                            selected_option
                        ),
                        correct=correct,
                        response_time_seconds=(
                            response_time
                        ),
                        predicted_probability=(
                            predicted_probability
                        ),
                        mastery_before=(
                            mastery_before
                        ),
                        mastery_after=(
                            mastery_after
                        ),
                        model_name=(
                            interaction_model_name
                        ),
                        dkt_probability=(
                            research_prediction.get(
                                "dkt_probability"
                            )
                        ),
                        sakt_probability=(
                            research_prediction.get(
                                "sakt_probability"
                            )
                        ),
                        research_prediction_status=(
                            research_prediction.get(
                                "status",
                                "not_recorded",
                            )
                        ),
                        research_data_status=(
                            research_prediction.get(
                                "data_status"
                            )
                        ),
                    )

                    st.session_state.mastery[
                        skill_id
                    ] = mastery_after

                    st.session_state.interaction_position = (
                        interaction_position
                    )

                    st.session_state.feedback = {
                        "correct": correct,
                        "selected_option": (
                            selected_option
                        ),
                        "mastery_before": (
                            mastery_before
                        ),
                        "mastery_after": (
                            mastery_after
                        ),
                        "predicted_probability": (
                            predicted_probability
                        ),
                        "research_prediction": (
                            research_prediction
                        ),
                    }

                    st.rerun()

        else:
            feedback = (
                st.session_state.feedback
            )

            correct_option = question[
                "correct_option"
            ]

            if feedback["correct"]:
                st.success("Correct answer.")
            else:
                st.error("Incorrect answer.")

            st.write(
                "**Correct answer:** "
                f"{correct_option}. "
                f"{option_text[correct_option]}"
            )

            st.info(question["explanation"])

            predicted_correctness = feedback[
                "predicted_probability"
            ]

            updated_mastery = feedback[
                "mastery_after"
            ]

            previous_mastery = feedback[
                "mastery_before"
            ]

            mastery_change = (
                updated_mastery
                - previous_mastery
            )

            metric_one, metric_two = (
                st.columns(2)
            )

            metric_one.metric(
                "BKT predicted correctness",
                f"{predicted_correctness:.1%}",
            )

            metric_two.metric(
                "Updated mastery",
                f"{updated_mastery:.1%}",
                delta=f"{mastery_change:+.1%}",
            )

            research_prediction = feedback.get(
                "research_prediction",
                {
                    "available": False,
                    "reason": (
                        "No optional research prediction "
                        "was generated for this question."
                    ),
                },
            )

            with st.expander(
                "Optional DKT and SAKT research predictions"
            ):
                st.caption(
                    "These estimates were generated before the "
                    "answer was recorded. They do not affect "
                    "question selection or mastery updates."
                )

                if research_prediction.get(
                    "available",
                    False,
                ):
                    (
                        bkt_column,
                        dkt_column,
                        sakt_column,
                    ) = st.columns(3)

                    bkt_column.metric(
                        "BKT (adaptive)",
                        f"{predicted_correctness:.1%}",
                    )

                    dkt_column.metric(
                        "DKT (research)",
                        (
                            f"{research_prediction['dkt_probability']:.1%}"
                        ),
                    )

                    sakt_column.metric(
                        "SAKT (research)",
                        (
                            f"{research_prediction['sakt_probability']:.1%}"
                        ),
                    )

                    current_probabilities = [
                        predicted_correctness,
                        research_prediction[
                            "dkt_probability"
                        ],
                        research_prediction[
                            "sakt_probability"
                        ],
                    ]
                    current_spread = (
                        max(current_probabilities)
                        - min(current_probabilities)
                    )

                    if current_spread >= 0.20:
                        st.warning(
                            "High model disagreement: the "
                            "probabilities span "
                            f"{current_spread:.1%}."
                        )
                    elif current_spread >= 0.10:
                        st.info(
                            "Moderate model disagreement: the "
                            "probabilities span "
                            f"{current_spread:.1%}."
                        )
                    else:
                        st.caption(
                            "Low model disagreement: the "
                            "probabilities span "
                            f"{current_spread:.1%}."
                        )

                    st.caption(
                        "Completed interactions used: "
                        f"{research_prediction['completed_interactions_used']}"
                    )

                    st.warning(
                        "DKT and SAKT were fitted using simulated "
                        "SFLA interactions. Their outputs are "
                        "technical research predictions, not "
                        "validated educational recommendations."
                    )
                else:
                    st.info(
                        research_prediction.get(
                            "reason",
                            "Research predictions are unavailable.",
                        )
                    )

            next_button_key = (
                "next_"
                f"{question['item_id']}"
            )

            if st.button(
                "Next question",
                type="primary",
                key=next_button_key,
            ):
                attempted_items = set(
                    st.session_state
                    .attempted_item_ids
                )

                attempted_items.add(
                    question["item_id"]
                )

                st.session_state.attempted_item_ids = (
                    attempted_items
                )

                answer_widget_key = (
                    "answer_"
                    f"{question['item_id']}"
                )

                if (
                    answer_widget_key
                    in st.session_state
                ):
                    del st.session_state[
                        answer_widget_key
                    ]

                st.session_state.current_question = (
                    None
                )

                st.session_state.feedback = None

                st.session_state.question_started_at = (
                    None
                )

                st.session_state.current_research_prediction = (
                    None
                )

                st.session_state.research_prediction_key = (
                    None
                )

                st.rerun()


with progress_tab:
    st.subheader(
        "Knowledge-component mastery"
    )

    mastery_rows = []

    for skill_id, skill_name in (
        SKILL_NAMES.items()
    ):
        mastery_rows.append(
            {
                "skill_id": skill_id,
                "knowledge_component": (
                    skill_name
                ),
                "mastery": (
                    st.session_state.mastery[
                        skill_id
                    ]
                ),
            }
        )

    mastery_table = pd.DataFrame(
        mastery_rows
    )

    st.bar_chart(
        mastery_table.set_index(
            "skill_id"
        )["mastery"],
        height=350,
    )

    display_mastery = mastery_table.copy()

    display_mastery["mastery"] = (
        display_mastery["mastery"].map(
            lambda value: f"{value:.1%}"
        )
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
            "Complete a question to generate "
            "progress data."
        )
    else:
        total_attempts = len(history)

        overall_accuracy = (
            history["actual"].mean()
        )

        weakest_skill = min(
            st.session_state.mastery,
            key=(
                st.session_state.mastery.get
            ),
        )

        weakest_skill_name = (
            SKILL_NAMES[weakest_skill]
        )

        (
            column_one,
            column_two,
            column_three,
        ) = st.columns(3)

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
            "Recommended next topic: "
            f"**{weakest_skill_name}**"
        )

        mastery_trajectory = (
            build_mastery_trajectory(
                history=history,
                initial_mastery=(
                    initial_mastery_by_skill
                ),
            )
        )

        st.subheader("Mastery over time")
        st.caption(
            "Each line shows the BKT mastery estimate "
            "after every completed interaction."
        )
        st.line_chart(
            mastery_trajectory.set_index(
                "attempt"
            ),
            height=420,
        )

        recent_history = (
            history.tail(10).copy()
        )

        recent_history["actual"] = (
            recent_history["actual"].map(
                {
                    1: "Correct",
                    0: "Incorrect",
                }
            )
        )

        st.subheader("Recent interactions")

        st.dataframe(
            recent_history,
            use_container_width=True,
            hide_index=True,
        )


with research_tab:
    st.subheader(
        "Research data export"
    )

    st.info(
        "The standard export is restricted to the "
        "active pseudonymous learner. Application-wide "
        "data is available only through the protected "
        "administrator section below."
    )

    all_interactions = (
        load_learner_interactions(
            active_learner
        )
    )

    all_mastery = (
        load_learner_mastery_table(
            active_learner
        )
    )

    common_evaluation_table = (
        build_common_prediction_table(
            all_interactions
        )
    )

    if all_interactions.empty:
        st.warning(
            "No interactions have been recorded "
            "for the active learner."
        )
    else:
        total_interactions = len(
            all_interactions
        )

        total_learners = (
            all_interactions[
                "learner_id"
            ].nunique()
        )

        represented_skills = (
            all_interactions[
                "skill_id"
            ].nunique()
        )

        platform_accuracy = (
            all_interactions[
                "actual"
            ].mean()
        )

        (
            export_column_one,
            export_column_two,
            export_column_three,
            export_column_four,
        ) = st.columns(4)

        export_column_one.metric(
            "Recorded interactions",
            total_interactions,
        )

        export_column_two.metric(
            "Learners",
            total_learners,
        )

        export_column_three.metric(
            "Skills represented",
            (
                f"{represented_skills}"
                f"/{len(SKILL_NAMES)}"
            ),
        )

        export_column_four.metric(
            "Observed accuracy",
            f"{platform_accuracy:.1%}",
        )

        st.subheader(
            "Interaction preview"
        )

        interaction_preview = (
            all_interactions
            .tail(20)
            .copy()
        )

        interaction_preview[
            "actual"
        ] = interaction_preview[
            "actual"
        ].map(
            {
                1: "Correct",
                0: "Incorrect",
            }
        )

        st.dataframe(
            interaction_preview,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Model disagreement analysis"
        )

        if common_evaluation_table.empty:
            st.info(
                "No interaction currently has predictions "
                "from all three models. Enable DKT and SAKT, "
                "then complete at least two questions."
            )
        else:
            high_disagreement_count = int(
                (
                    common_evaluation_table[
                        "disagreement_level"
                    ]
                    == "High"
                ).sum()
            )

            (
                model_column_one,
                model_column_two,
                model_column_three,
            ) = st.columns(3)

            model_column_one.metric(
                "Common prediction rows",
                len(common_evaluation_table),
            )
            model_column_two.metric(
                "Mean probability spread",
                (
                    f"{common_evaluation_table['probability_spread'].mean():.1%}"
                ),
            )
            model_column_three.metric(
                "High-disagreement rows",
                high_disagreement_count,
            )

            st.caption(
                "This strict like-for-like table contains only "
                "interactions for which BKT, DKT and SAKT all "
                "produced a probability before the response. "
                "High disagreement means a probability spread "
                "of at least 20 percentage points."
            )

            st.dataframe(
                common_evaluation_table.tail(20),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader(
            "Current mastery states"
        )

        mastery_preview = (
            all_mastery.copy()
        )

        if not mastery_preview.empty:
            mastery_preview[
                "mastery"
            ] = mastery_preview[
                "mastery"
            ].map(
                lambda value: (
                    f"{value:.1%}"
                )
            )

        st.dataframe(
            mastery_preview,
            use_container_width=True,
            hide_index=True,
        )

        export_timestamp = (
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%SZ"
            )
        )

        interaction_csv = (
            all_interactions
            .to_csv(index=False)
            .encode("utf-8")
        )

        mastery_csv = (
            all_mastery
            .to_csv(index=False)
            .encode("utf-8")
        )

        common_evaluation_csv = (
            common_evaluation_table
            .to_csv(index=False)
            .encode("utf-8")
        )

        export_manifest = {
            "exported_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "export_scope": "active_learner_only",
            "domain": "sfla",
            "model": "BKT",
            "adaptive_engine": "BKT",
            "optional_research_models": [
                "DKT",
                "Causal SAKT",
            ],
            "research_models_control_recommendations": (
                False
            ),
            "parameter_data_status": (
                parameter_data_status
            ),
            "interaction_rows": int(
                total_interactions
            ),
            "unique_learners": int(
                total_learners
            ),
            "represented_skills": int(
                represented_skills
            ),
            "question_bank_size": int(
                len(question_bank)
            ),
            "common_prediction_rows": int(
                len(common_evaluation_table)
            ),
            "high_disagreement_rows": int(
                (
                    common_evaluation_table[
                        "disagreement_level"
                    ]
                    == "High"
                ).sum()
                if not common_evaluation_table.empty
                else 0
            ),
            "validity_interpretation": (
                bkt_artifact[
                    "validity_interpretation"
                ]
            ),
            "privacy_note": (
                "This export contains only the active "
                "pseudonymous learner."
            ),
        }

        manifest_json = json.dumps(
            export_manifest,
            indent=2,
        ).encode("utf-8")

        st.subheader(
            "Download research artifacts"
        )

        (
            download_column_one,
            download_column_two,
            download_column_three,
            download_column_four,
        ) = st.columns(4)

        download_column_one.download_button(
            label="Download interactions",
            data=interaction_csv,
            file_name=(
                "sfla_learner_interactions_"
                f"{export_timestamp}.csv"
            ),
            mime="text/csv",
        )

        download_column_two.download_button(
            label="Download mastery states",
            data=mastery_csv,
            file_name=(
                "sfla_learner_mastery_"
                f"{export_timestamp}.csv"
            ),
            mime="text/csv",
        )

        download_column_three.download_button(
            label="Download export manifest",
            data=manifest_json,
            file_name=(
                "sfla_learner_manifest_"
                f"{export_timestamp}.json"
            ),
            mime="application/json",
        )

        download_column_four.download_button(
            label="Download disagreement analysis",
            data=common_evaluation_csv,
            file_name=(
                "sfla_learner_disagreement_"
                f"{export_timestamp}.csv"
            ),
            mime="text/csv",
            disabled=common_evaluation_table.empty,
        )

    st.warning(
        "Storage notice: this demonstration uses a local "
        "SQLite database. On Streamlit Community Cloud, "
        "records may be lost when the application restarts "
        "or is redeployed. Use a managed database and an "
        "approved retention process before collecting real "
        "participant data."
    )

    if "admin_export_unlocked" not in st.session_state:
        st.session_state.admin_export_unlocked = False

    with st.expander(
        "Administrator export (all learners)"
    ):
        administrator_password = (
            get_admin_password()
        )

        if not administrator_password:
            st.info(
                "Application-wide export is disabled. "
                "Configure `SFLA_ADMIN_PASSWORD` in "
                "Streamlit Secrets to enable it."
            )
        else:
            if not st.session_state[
                "admin_export_unlocked"
            ]:
                with st.form(
                    "administrator_export_login",
                    clear_on_submit=True,
                ):
                    supplied_password = st.text_input(
                        "Administrator password",
                        type="password",
                    )
                    unlock_administrator_export = (
                        st.form_submit_button(
                            "Unlock administrator export"
                        )
                    )

                if unlock_administrator_export:
                    if hmac.compare_digest(
                        supplied_password,
                        administrator_password,
                    ):
                        st.session_state[
                            "admin_export_unlocked"
                        ] = True
                        st.rerun()
                    else:
                        st.error(
                            "The administrator password "
                            "is incorrect."
                        )

            if st.session_state[
                "admin_export_unlocked"
            ]:
                st.success(
                    "Administrator export is unlocked "
                    "for this browser session."
                )

                if st.button(
                    "Lock administrator export",
                    key="lock_administrator_export",
                ):
                    st.session_state[
                        "admin_export_unlocked"
                    ] = False
                    st.rerun()

                administrator_interactions = (
                    load_all_interactions()
                )
                administrator_mastery = (
                    load_all_mastery()
                )

                render_research_export_scope(
                    interactions=(
                        administrator_interactions
                    ),
                    mastery=administrator_mastery,
                    scope_label=(
                        "Administrator: all learners"
                    ),
                    file_prefix=(
                        "sfla_administrator"
                    ),
                    parameter_status=(
                        parameter_data_status
                    ),
                    question_bank_size=len(
                        question_bank
                    ),
                    validity_interpretation=(
                        bkt_artifact[
                            "validity_interpretation"
                        ]
                    ),
                )


with about_tab:
    st.subheader("About this prototype")

    st.write(
        """
        The platform applies per-skill Bayesian
        Knowledge Tracing parameters exported from
        the SFLA cross-domain experiment.

        Each response updates the mastery estimate
        for its associated knowledge component, and
        the recommendation engine prioritises
        lower-mastery skills.

        BKT is the only adaptive engine. Optional DKT
        and causal SAKT probabilities can be displayed
        and exported for research comparison, but they
        never influence question selection or mastery
        updates.
        """
    )

    st.write(
        "**Parameter data status:** "
        f"{parameter_data_status}"
    )

    validity_interpretation = bkt_artifact[
        "validity_interpretation"
    ]

    st.write(
        "**Validity interpretation:** "
        f"{validity_interpretation}"
    )

    st.subheader("Privacy and data persistence")

    st.write(
        "Use pseudonymous learner IDs only. Do not enter "
        "names, email addresses, student numbers or other "
        "personal identifiers. Standard exports contain "
        "only the active learner; application-wide exports "
        "require the administrator password."
    )

    st.warning(
        "The demonstration stores interactions in a local "
        "SQLite database. Hosted records can be reset when "
        "the application restarts or is redeployed. A managed "
        "persistent database, participant information, consent "
        "and a retention schedule are required before using "
        "the platform with real participants."
    )
