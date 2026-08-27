from datetime import datetime, timezone
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
    load_learner_mastery,
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
            "Learner ID",
            placeholder="For example: learner_001",
        )

        start_session = (
            st.form_submit_button(
                "Start session",
                type="primary",
            )
        )

    if start_session:
        learner_id = learner_input.strip()

        if not learner_id:
            st.error("Enter a learner ID.")
        else:
            st.session_state.learner_id = (
                learner_id
            )

            st.session_state.session_id = str(
                uuid.uuid4()
            )

            st.session_state.mastery = (
                load_learner_mastery(
                    learner_id=learner_id,
                    initial_mastery_by_skill=(
                        initial_mastery_by_skill
                    ),
                )
            )

            st.session_state.attempted_item_ids = (
                set()
            )

            st.session_state.current_question = (
                None
            )

            st.session_state.feedback = None

            st.session_state.interaction_position = (
                0
            )

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

    if st.session_state.current_question is None:
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
        st.session_state.current_question
    )

    if question is None:
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
        "Use pseudonymous learner IDs only. "
        "Do not enter learner names, email "
        "addresses or student numbers."
    )

    all_interactions = (
        load_all_interactions()
    )

    all_mastery = (
        load_all_mastery()
    )

    common_evaluation_table = (
        all_interactions.dropna(
            subset=[
                "predicted_probability",
                "dkt_probability",
                "sakt_probability",
            ]
        )[
            [
                "learner_id",
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

    if all_interactions.empty:
        st.warning(
            "No platform interactions have "
            "been recorded."
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
            "Common prediction subset"
        )

        if common_evaluation_table.empty:
            st.info(
                "No interaction currently has predictions "
                "from all three models. Enable DKT and SAKT, "
                "then complete at least two questions."
            )
        else:
            st.caption(
                "This strict like-for-like table contains only "
                "interactions for which BKT, DKT and SAKT all "
                "produced a probability before the response."
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
            "validity_interpretation": (
                bkt_artifact[
                    "validity_interpretation"
                ]
            ),
            "privacy_note": (
                "Learner IDs must be "
                "pseudonymous."
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
                "sfla_platform_interactions_"
                f"{export_timestamp}.csv"
            ),
            mime="text/csv",
        )

        download_column_two.download_button(
            label="Download mastery states",
            data=mastery_csv,
            file_name=(
                "sfla_platform_mastery_"
                f"{export_timestamp}.csv"
            ),
            mime="text/csv",
        )

        download_column_three.download_button(
            label="Download export manifest",
            data=manifest_json,
            file_name=(
                "sfla_platform_manifest_"
                f"{export_timestamp}.json"
            ),
            mime="application/json",
        )

        download_column_four.download_button(
            label="Download common predictions",
            data=common_evaluation_csv,
            file_name=(
                "sfla_platform_common_predictions_"
                f"{export_timestamp}.csv"
            ),
            mime="text/csv",
            disabled=common_evaluation_table.empty,
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