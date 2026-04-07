"""Role-specific span length priors."""

ROLE_LENGTH_PRIORS = {
    "ARG0": (1, 8),
    "ARG1": (1, 12),
    "ARG2": (1, 10),
    "ARGM-LOC": (1, 8),
    "ARGM-TMP": (1, 7),
    "ARGM-MNR": (1, 8),
    "ARGM-CAU": (2, 14),
    "ARGM-PRP": (2, 14),
}


def length_penalty(role: str, span_length: int) -> float:
    low, high = ROLE_LENGTH_PRIORS.get(role, (1, 18))
    if span_length < low:
        return 0.10 * (low - span_length)
    if span_length > high:
        return 0.05 * (span_length - high)
    return 0.0
