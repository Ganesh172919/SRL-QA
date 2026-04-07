"""QA-Noun-style templates for nominal semantic coverage."""

NOMINAL_TEMPLATES = {
    "agent": "Who caused or performed the {noun}?",
    "patient": "What was affected by the {noun}?",
    "location": "Where did the {noun} happen?",
    "time": "When did the {noun} happen?",
    "purpose": "Why did the {noun} happen?",
}


def generate_nominal_questions(noun: str) -> list[dict[str, str]]:
    return [
        {"nominal_role": role, "question": template.format(noun=noun)}
        for role, template in NOMINAL_TEMPLATES.items()
    ]
