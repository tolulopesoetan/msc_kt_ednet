from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pandas as pd

from .config import DATABASE_PATH


def initialise_database(
    database_path=DATABASE_PATH,
):
    database_path = Path(database_path)

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(
        database_path
    ) as connection:
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
                model_name TEXT NOT NULL,
                dkt_probability REAL,
                sakt_probability REAL,
                research_prediction_status TEXT NOT NULL
                    DEFAULT 'not_recorded',
                research_data_status TEXT
            )
            """
        )

        # Upgrade databases created before DKT and SAKT
        # research predictions were introduced.
        existing_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(interactions)"
            ).fetchall()
        }

        optional_research_columns = {
            "dkt_probability": "REAL",
            "sakt_probability": "REAL",
            "research_prediction_status": (
                "TEXT NOT NULL DEFAULT 'not_recorded'"
            ),
            "research_data_status": "TEXT",
        }

        for column_name, column_definition in (
            optional_research_columns.items()
        ):
            if column_name not in existing_columns:
                connection.execute(
                    "ALTER TABLE interactions "
                    f"ADD COLUMN {column_name} "
                    f"{column_definition}"
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


def load_learner_mastery(
    learner_id,
    initial_mastery_by_skill,
    database_path=DATABASE_PATH,
):
    mastery = {
        skill_id: float(initial_mastery)
        for skill_id, initial_mastery
        in initial_mastery_by_skill.items()
    }

    with sqlite3.connect(
        database_path
    ) as connection:
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
            mastery[skill_id] = float(
                mastery_value
            )

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
    model_name="BKT",
    database_path=DATABASE_PATH,
    dkt_probability=None,
    sakt_probability=None,
    research_prediction_status=(
        "not_recorded"
    ),
    research_data_status=None,
):
    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with sqlite3.connect(
        database_path
    ) as connection:
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
                model_name,
                dkt_probability,
                sakt_probability,
                research_prediction_status,
                research_data_status
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
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
                model_name,
                (
                    None
                    if dkt_probability is None
                    else float(dkt_probability)
                ),
                (
                    None
                    if sakt_probability is None
                    else float(sakt_probability)
                ),
                research_prediction_status,
                research_data_status,
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


def load_interaction_history(
    learner_id,
    database_path=DATABASE_PATH,
):
    with sqlite3.connect(
        database_path
    ) as connection:
        history = pd.read_sql_query(
            """
            SELECT
                timestamp,
                session_id,
                interaction_position,
                item_id,
                skill_id,
                selected_option,
                correct_option,
                actual,
                predicted_probability,
                mastery_before,
                mastery_after,
                response_time_seconds,
                model_name,
                dkt_probability,
                sakt_probability,
                research_prediction_status,
                research_data_status
            FROM interactions
            WHERE learner_id = ?
            ORDER BY id
            """,
            connection,
            params=(learner_id,),
        )

    return history


def load_learner_interactions(
    learner_id,
    database_path=DATABASE_PATH,
):
    """Return export-ready interactions for one learner only."""

    with sqlite3.connect(
        database_path
    ) as connection:
        interactions = pd.read_sql_query(
            """
            SELECT
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
                model_name,
                dkt_probability,
                sakt_probability,
                research_prediction_status,
                research_data_status
            FROM interactions
            WHERE learner_id = ?
            ORDER BY id
            """,
            connection,
            params=(learner_id,),
        )

    return interactions


def load_session_history(
    learner_id,
    session_id,
    database_path=DATABASE_PATH,
):
    """Return completed interactions from one learner session."""

    with sqlite3.connect(
        database_path
    ) as connection:
        history = pd.read_sql_query(
            """
            SELECT
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
                model_name,
                dkt_probability,
                sakt_probability,
                research_prediction_status,
                research_data_status
            FROM interactions
            WHERE learner_id = ?
              AND session_id = ?
            ORDER BY id
            """,
            connection,
            params=(learner_id, session_id),
        )

    return history


def load_learner_mastery_table(
    learner_id,
    database_path=DATABASE_PATH,
):
    """Return persisted mastery rows for one learner only."""

    with sqlite3.connect(
        database_path
    ) as connection:
        mastery = pd.read_sql_query(
            """
            SELECT
                learner_id,
                skill_id,
                mastery,
                updated_at
            FROM learner_mastery
            WHERE learner_id = ?
            ORDER BY skill_id
            """,
            connection,
            params=(learner_id,),
        )

    return mastery


def load_all_interactions(
    database_path=DATABASE_PATH,
):
    with sqlite3.connect(
        database_path
    ) as connection:
        interactions = pd.read_sql_query(
            """
            SELECT
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
                model_name,
                dkt_probability,
                sakt_probability,
                research_prediction_status,
                research_data_status
            FROM interactions
            ORDER BY id
            """,
            connection,
        )

    return interactions


def load_all_mastery(
    database_path=DATABASE_PATH,
):
    with sqlite3.connect(
        database_path
    ) as connection:
        mastery = pd.read_sql_query(
            """
            SELECT
                learner_id,
                skill_id,
                mastery,
                updated_at
            FROM learner_mastery
            ORDER BY learner_id, skill_id
            """,
            connection,
        )

    return mastery
