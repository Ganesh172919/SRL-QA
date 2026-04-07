"""Interactive terminal runner for all SRL-QA models in the workspace."""

from __future__ import annotations

import argparse

from srlqa.model_hub import MODEL_SPECS, ModelHub, model_choices, model_labels


def print_menu() -> None:
    labels = model_labels(include_all=True)
    print("\nAvailable model choices:")
    for index, key in enumerate(model_choices(include_all=True), start=1):
        print(f"{index}. {labels[key]} [{key}]")
    print("\nModel notes:")
    for spec in MODEL_SPECS:
        print(f"- {spec.label}: {spec.description}")


def choose_model() -> str:
    choices = model_choices(include_all=True)
    labels = model_labels(include_all=True)
    while True:
        value = input("\nSelect model number or key: ").strip()
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return choices[int(value) - 1]
        if value in choices:
            return value
        print("Invalid choice. Try again.")
        print("Valid keys:", ", ".join(f"{labels[key]}={key}" for key in choices))


def print_result(result: dict[str, object]) -> None:
    print("\n" + "=" * 72)
    print(f"Model: {result['model_label']} ({result['model_key']})")
    print(f"OK: {result['ok']} | Latency: {float(result['latency_ms']):.1f} ms")
    if not result["ok"]:
        print(f"Error: {result['error']}")
        return
    print(f"Answer: {result['answer']}")
    print(f"Role: {result['role']}")
    print(f"Confidence: {float(result['confidence']):.4f}")
    print(f"Reasoning: {result['reasoning']}")


def run_once(model_key: str, context: str, question: str, expected_answer: str | None = None) -> None:
    """Run one non-interactive all-model or single-model query."""

    hub = ModelHub()
    print("RAISE/Hybrid SRL-QA All-Model Runner")
    print(f"Selected: {model_labels(include_all=True)[model_key]} [{model_key}]")
    print(f"Context: {context}")
    print(f"Question: {question}")
    if expected_answer:
        print(f"Expected answer: {expected_answer}")
    for result in hub.run(model_key, context, question, expected_answer=expected_answer):
        print_result(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more SRL-QA model families.")
    parser.add_argument(
        "--model",
        choices=model_choices(include_all=True),
        default=None,
        help="Model key to run. Use 'all' to compare every available model.",
    )
    parser.add_argument("--context", help="Input context sentence or paragraph.")
    parser.add_argument("--question", help="Question to ask about the context.")
    parser.add_argument("--expected-answer", help="Optional gold answer for correction/evaluation mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context or args.question:
        if not args.context or not args.question:
            raise SystemExit("--context and --question must be provided together.")
        run_once(args.model or "all", args.context, args.question, args.expected_answer)
        return

    print("RAISE/Hybrid SRL-QA All-Model Runner")
    print("Paste a context, ask a question, and compare all available model families.")
    print_menu()
    hub = ModelHub()

    while True:
        model_key = choose_model()
        context = input("\nContext (or 'quit'): ").strip()
        if context.lower() in {"quit", "exit", "q"}:
            break
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        expected = input("Expected answer for correction/eval (optional): ").strip() or None

        for result in hub.run(model_key, context, question, expected_answer=expected):
            print_result(result)

        again = input("\nAsk another? [y/N]: ").strip().lower()
        if again not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()
