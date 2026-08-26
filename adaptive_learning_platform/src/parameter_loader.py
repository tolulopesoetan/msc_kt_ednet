import json
import math
from pathlib import Path

from .config import (
    BKT_PARAMETERS_PATH,
    SKILL_NAMES,
)


REQUIRED_BKT_PARAMETERS = {
    "initial_mastery",
    "learning_probability",
    "guess_probability",
    "slip_probability",
    "forget_probability",
}


def load_bkt_parameter_artifact(
    parameter_path=BKT_PARAMETERS_PATH,
):
    parameter_path = Path(parameter_path)

    if not parameter_path.exists():
        raise FileNotFoundError(
            "SFLA BKT parameter file not found: "
            f"{parameter_path}"
        )

    try:
        with parameter_path.open(
            "r",
            encoding="utf-8",
        ) as parameter_file:
            artifact = json.load(parameter_file)

    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid JSON in BKT parameter file: "
            f"{error}"
        ) from error

    required_artifact_fields = {
        "schema_version",
        "model",
        "domain",
        "data_status",
        "validity_interpretation",
        "skills",
    }

    missing_artifact_fields = (
        required_artifact_fields
        - set(artifact)
    )

    if missing_artifact_fields:
        raise ValueError(
            "BKT parameter artifact is missing fields: "
            f"{sorted(missing_artifact_fields)}"
        )

    if artifact["schema_version"] != 1:
        raise ValueError(
            "Unsupported BKT parameter schema version: "
            f"{artifact['schema_version']}"
        )

    if str(artifact["model"]).upper() != "BKT":
        raise ValueError(
            "The parameter artifact is not a BKT model."
        )

    if str(artifact["domain"]).lower() != "sfla":
        raise ValueError(
            "The parameter artifact is not for SFLA."
        )

    skill_parameters = artifact["skills"]

    if not isinstance(skill_parameters, dict):
        raise ValueError(
            "The artifact's skills field must "
            "be an object."
        )

    expected_skills = set(SKILL_NAMES)
    observed_skills = set(skill_parameters)

    missing_skills = (
        expected_skills
        - observed_skills
    )

    unexpected_skills = (
        observed_skills
        - expected_skills
    )

    if missing_skills:
        raise ValueError(
            "BKT parameters are missing skills: "
            f"{sorted(missing_skills)}"
        )

    if unexpected_skills:
        raise ValueError(
            "BKT parameters contain unexpected skills: "
            f"{sorted(unexpected_skills)}"
        )

    normalised_skill_parameters = {}

    for skill_id in sorted(expected_skills):
        parameters = skill_parameters[skill_id]

        if not isinstance(parameters, dict):
            raise ValueError(
                f"Parameters for {skill_id} "
                "must be an object."
            )

        missing_parameters = (
            REQUIRED_BKT_PARAMETERS
            - set(parameters)
        )

        if missing_parameters:
            raise ValueError(
                f"{skill_id} is missing BKT parameters: "
                f"{sorted(missing_parameters)}"
            )

        normalised_parameters = {}

        for parameter_name in (
            REQUIRED_BKT_PARAMETERS
        ):
            try:
                parameter_value = float(
                    parameters[parameter_name]
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{skill_id} {parameter_name} "
                    "must be numeric."
                ) from error

            if not math.isfinite(parameter_value):
                raise ValueError(
                    f"{skill_id} {parameter_name} "
                    "must be finite."
                )

            if not 0 <= parameter_value <= 1:
                raise ValueError(
                    f"{skill_id} {parameter_name} "
                    "must be between zero and one."
                )

            normalised_parameters[
                parameter_name
            ] = parameter_value

        normalised_skill_parameters[
            skill_id
        ] = normalised_parameters

    validated_artifact = dict(artifact)

    validated_artifact["data_status"] = str(
        artifact["data_status"]
    ).upper()

    validated_artifact["skills"] = (
        normalised_skill_parameters
    )

    return validated_artifact