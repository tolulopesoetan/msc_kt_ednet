from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "sfla"
    / "sfla_neural_metadata.json"
)


class ResearchModelError(RuntimeError):
    """Raised when optional research models cannot be used."""


@dataclass(frozen=True)
class ResearchModelBundle:
    dkt_model: Any
    sakt_model: Any
    metadata: dict[str, Any]
    project_root: Path


def _import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ResearchModelError(
            "TensorFlow is not installed. BKT can continue to "
            "operate, but DKT and SAKT predictions are unavailable."
        ) from error

    return tf


def _extract_model_output(output: Any) -> np.ndarray:
    if isinstance(output, dict):
        if not output:
            raise ResearchModelError(
                "The research model returned an empty output."
            )

        output = next(iter(output.values()))

    output_array = np.asarray(output)

    if output_array.ndim != 3:
        raise ResearchModelError(
            "Unexpected research-model output shape: "
            f"{output_array.shape}"
        )

    return output_array


def _serve_probability(
    model: Any,
    interaction_tokens: np.ndarray,
    target_skill_tokens: np.ndarray,
) -> float:
    tf = _import_tensorflow()

    output = model.serve(
        interaction_tokens=tf.convert_to_tensor(
            interaction_tokens,
            dtype=tf.int32,
        ),
        target_skill_tokens=tf.convert_to_tensor(
            target_skill_tokens,
            dtype=tf.int32,
        ),
    )

    output_array = _extract_model_output(output)
    probability = float(output_array[0, -1, 0])

    if not np.isfinite(probability):
        raise ResearchModelError(
            "The research model returned a non-finite probability."
        )

    if probability < 0.0 or probability > 1.0:
        raise ResearchModelError(
            "The research model returned a probability outside "
            f"[0, 1]: {probability}"
        )

    return probability


def _validate_reference_probe(
    bundle: ResearchModelBundle,
) -> None:
    probe = bundle.metadata.get("reference_probe")

    if not probe:
        raise ResearchModelError(
            "The neural metadata does not contain a reference probe."
        )

    interaction_tokens = np.asarray(
        [probe["interaction_tokens"]],
        dtype=np.int32,
    )

    target_skill_tokens = np.asarray(
        [probe["target_skill_tokens"]],
        dtype=np.int32,
    )

    dkt_probability = _serve_probability(
        bundle.dkt_model,
        interaction_tokens,
        target_skill_tokens,
    )

    sakt_probability = _serve_probability(
        bundle.sakt_model,
        interaction_tokens,
        target_skill_tokens,
    )

    if not np.isclose(
        dkt_probability,
        float(probe["expected_dkt_probability"]),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ResearchModelError(
            "The loaded DKT artifact failed its reference check."
        )

    if not np.isclose(
        sakt_probability,
        float(probe["expected_sakt_probability"]),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ResearchModelError(
            "The loaded SAKT artifact failed its reference check."
        )


def load_research_models(
    project_root: str | Path | None = None,
) -> ResearchModelBundle:
    """
    Load and verify the optional SFLA DKT and SAKT artifacts.

    These models provide research predictions only. They do not
    control question selection or BKT mastery updates.
    """

    tf = _import_tensorflow()

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else PROJECT_ROOT
    )

    metadata_path = (
        root
        / "models"
        / "sfla"
        / "sfla_neural_metadata.json"
    )

    if not metadata_path.exists():
        raise ResearchModelError(
            f"Neural metadata not found: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        metadata = json.load(metadata_file)

    roles = metadata.get("platform_roles", {})

    if roles.get("adaptive_engine") != "BKT":
        raise ResearchModelError(
            "Invalid metadata: BKT must remain the adaptive engine."
        )

    if roles.get(
        "research_models_control_recommendations",
        True,
    ):
        raise ResearchModelError(
            "Invalid metadata: research models must not control "
            "recommendations."
        )

    expected_tensorflow_version = metadata.get(
        "tensorflow_version"
    )

    if (
        expected_tensorflow_version
        and tf.__version__ != expected_tensorflow_version
    ):
        raise ResearchModelError(
            "TensorFlow version mismatch. "
            f"Artifact: {expected_tensorflow_version}; "
            f"platform: {tf.__version__}."
        )

    dkt_path = (
        root
        / metadata["models"]["DKT"][
            "artifact_directory"
        ]
    )

    sakt_path = (
        root
        / metadata["models"]["Causal SAKT"][
            "artifact_directory"
        ]
    )

    if not dkt_path.exists():
        raise ResearchModelError(
            f"DKT artifact not found: {dkt_path}"
        )

    if not sakt_path.exists():
        raise ResearchModelError(
            f"SAKT artifact not found: {sakt_path}"
        )

    dkt_model = tf.saved_model.load(
        str(dkt_path)
    )

    sakt_model = tf.saved_model.load(
        str(sakt_path)
    )

    bundle = ResearchModelBundle(
        dkt_model=dkt_model,
        sakt_model=sakt_model,
        metadata=metadata,
        project_root=root,
    )

    _validate_reference_probe(bundle)

    return bundle


def load_research_models_safely(
    project_root: str | Path | None = None,
) -> tuple[ResearchModelBundle | None, str | None]:
    """
    Load optional models without stopping the BKT application.

    Returns:
        (bundle, None) when successful.
        (None, error_message) when unavailable.
    """

    try:
        bundle = load_research_models(
            project_root=project_root
        )
        return bundle, None
    except Exception as error:
        return None, str(error)


def _history_to_records(
    history: Iterable[Mapping[str, Any]] | Any,
) -> list[Mapping[str, Any]]:
    if history is None:
        return []

    if isinstance(history, Mapping):
        return [history]

    if hasattr(history, "to_dict"):
        try:
            return history.to_dict(
                orient="records"
            )
        except TypeError:
            pass

    return list(history)


def _get_record_value(
    record: Mapping[str, Any],
    possible_keys: tuple[str, ...],
) -> Any:
    for key in possible_keys:
        if key in record:
            return record[key]

    raise ResearchModelError(
        "Interaction history is missing one of the required "
        f"fields: {possible_keys}"
    )


def predict_research_probabilities(
    bundle: ResearchModelBundle,
    completed_history: Iterable[
        Mapping[str, Any]
    ] | Any,
    next_skill_id: str,
) -> dict[str, Any]:
    """
    Predict the next response using DKT and causal SAKT.

    completed_history must be in chronological order and must
    contain only questions already answered by the learner.
    The current question's response must never be included.
    """

    records = _history_to_records(
        completed_history
    )

    if not records:
        return {
            "available": False,
            "reason": (
                "At least one completed interaction is required "
                "before DKT and SAKT can produce a prediction."
            ),
            "dkt_probability": None,
            "sakt_probability": None,
            "model_role": "research_only",
            "controls_recommendations": False,
        }

    metadata = bundle.metadata
    sequence_contract = metadata[
        "sequence_contract"
    ]
    encoding = metadata["encoding"]

    sequence_length = int(
        sequence_contract["sequence_length"]
    )

    maximum_history = int(
        sequence_contract[
            "maximum_completed_history"
        ]
    )

    skill_to_index = encoding[
        "skill_to_zero_based_index"
    ]

    skill_to_target_token = encoding[
        "skill_to_target_token"
    ]

    next_skill_id = str(next_skill_id)

    if next_skill_id not in skill_to_target_token:
        raise ResearchModelError(
            f"Unknown target skill: {next_skill_id}"
        )

    recent_records = records[
        -maximum_history:
    ]

    encoded_interactions: list[int] = []

    for record in recent_records:
        skill_id = str(
            _get_record_value(
                record,
                (
                    "skill_id",
                    "skill_name",
                ),
            )
        )

        if skill_id not in skill_to_index:
            raise ResearchModelError(
                f"Unknown historical skill: {skill_id}"
            )

        correctness = int(
            _get_record_value(
                record,
                (
                    "correct",
                    "actual",
                    "is_correct",
                ),
            )
        )

        if correctness not in (0, 1):
            raise ResearchModelError(
                "Correctness values must be either 0 or 1."
            )

        skill_index = int(
            skill_to_index[skill_id]
        )

        interaction_token = (
            (2 * skill_index)
            + correctness
            + 1
        )

        encoded_interactions.append(
            interaction_token
        )

    interaction_tokens = np.zeros(
        (1, sequence_length),
        dtype=np.int32,
    )

    target_skill_tokens = np.zeros(
        (1, sequence_length),
        dtype=np.int32,
    )

    interaction_tokens[
        0,
        -len(encoded_interactions):,
    ] = encoded_interactions

    target_skill_tokens[0, -1] = int(
        skill_to_target_token[next_skill_id]
    )

    dkt_probability = _serve_probability(
        bundle.dkt_model,
        interaction_tokens,
        target_skill_tokens,
    )

    sakt_probability = _serve_probability(
        bundle.sakt_model,
        interaction_tokens,
        target_skill_tokens,
    )

    return {
        "available": True,
        "reason": None,
        "next_skill_id": next_skill_id,
        "completed_interactions_used": len(
            encoded_interactions
        ),
        "dkt_probability": dkt_probability,
        "sakt_probability": sakt_probability,
        "model_role": "research_only",
        "controls_recommendations": False,
        "data_status": metadata["data_status"],
        "validity_interpretation": metadata[
            "validity_interpretation"
        ],
    }