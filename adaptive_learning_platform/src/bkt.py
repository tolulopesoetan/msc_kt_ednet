def validate_bkt_parameters(parameters):
    required_parameters = {
        "initial_mastery",
        "learning_probability",
        "guess_probability",
        "slip_probability",
    }

    missing_parameters = (
        required_parameters - set(parameters)
    )

    if missing_parameters:
        raise ValueError(
            "Missing BKT parameters: "
            f"{sorted(missing_parameters)}"
        )

    for parameter_name in required_parameters:
        parameter_value = parameters[parameter_name]

        if not 0 <= parameter_value < 1:
            raise ValueError(
                f"{parameter_name} must be between "
                "zero and one."
            )


def clamp_probability(value):
    return min(
        max(float(value), 0.0001),
        0.9999,
    )


def probability_correct(
    prior_mastery,
    parameters,
):
    validate_bkt_parameters(parameters)

    prior_mastery = clamp_probability(
        prior_mastery
    )

    guess = parameters["guess_probability"]
    slip = parameters["slip_probability"]

    predicted_probability = (
        prior_mastery * (1 - slip)
        + (1 - prior_mastery) * guess
    )

    return clamp_probability(
        predicted_probability
    )


def update_mastery(
    prior_mastery,
    correct,
    parameters,
):
    validate_bkt_parameters(parameters)

    prior_mastery = clamp_probability(
        prior_mastery
    )

    learning = parameters[
        "learning_probability"
    ]
    guess = parameters[
        "guess_probability"
    ]
    slip = parameters[
        "slip_probability"
    ]

    if correct:
        numerator = (
            prior_mastery * (1 - slip)
        )

        denominator = numerator + (
            (1 - prior_mastery) * guess
        )
    else:
        numerator = (
            prior_mastery * slip
        )

        denominator = numerator + (
            (1 - prior_mastery)
            * (1 - guess)
        )

    posterior_mastery = (
        numerator / denominator
    )

    updated_mastery = (
        posterior_mastery
        + (1 - posterior_mastery)
        * learning
    )

    return clamp_probability(
        updated_mastery
    )