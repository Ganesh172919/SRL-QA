"""Minimal pairs for agency and patient diagnostics."""

MINIMAL_PAIRS = [
    {
        "context_a": "The company hired Rahul.",
        "context_b": "Rahul hired the company.",
        "question": "Who hired?",
        "role": "ARG0",
    },
    {
        "context_a": "The nurse treated the patient.",
        "context_b": "The patient treated the nurse.",
        "question": "Who was treated?",
        "role": "ARG1",
    },
]
