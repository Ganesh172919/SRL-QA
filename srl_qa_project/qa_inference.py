"""Inference utilities and demonstration script for PropQA-Net."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from config import ProjectConfig
from evaluator import load_trained_model, normalize_text


def simple_word_tokenize(text: str) -> List[str]:
    """Tokenize text into words and punctuation."""

    return re.findall(r"[A-Za-z0-9$%]+(?:[-'][A-Za-z0-9$%]+)*|[^\w\s]", text)


def simple_lemmatize(token: str) -> str:
    """Approximate a surface form lemma."""

    lower = token.lower()
    if lower.endswith("ing") and len(lower) > 4:
        return lower[:-3]
    if lower.endswith("ed") and len(lower) > 3:
        return lower[:-2]
    if lower.endswith("es") and len(lower) > 3:
        return lower[:-2]
    if lower.endswith("s") and len(lower) > 3:
        return lower[:-1]
    return lower


def heuristic_pos_tags(tokens: Sequence[str]) -> List[str]:
    """Assign lightweight POS tags for raw-text inference."""

    verb_lexicon = {
        "cook",
        "cooked",
        "send",
        "sent",
        "announce",
        "announced",
        "give",
        "gave",
        "deliver",
        "delivered",
        "arrive",
        "arrived",
        "sign",
        "signed",
        "purchase",
        "purchased",
        "repair",
        "repaired",
        "approve",
        "approved",
        "investigate",
        "investigated",
    }

    tags: List[str] = []
    for token in tokens:
        lower = token.lower()
        if re.fullmatch(r"[.,!?;:]", token):
            tags.append(token)
        elif lower in verb_lexicon or lower.endswith("ed") or lower.endswith("ing"):
            tags.append("VB")
        elif token.istitle():
            tags.append("NNP")
        elif token.isdigit():
            tags.append("CD")
        elif lower in {"in", "on", "at", "to", "for", "because", "by", "with"}:
            tags.append("IN")
        else:
            tags.append("NN")
    return tags


def infer_predicate_index(tokens: Sequence[str], question: str) -> int:
    """Infer the most likely predicate token for a context/question pair."""

    question_tokens = [simple_lemmatize(token) for token in simple_word_tokenize(question)]
    for index, token in enumerate(tokens):
        if simple_lemmatize(token) in question_tokens:
            return index

    for index, token in enumerate(tokens):
        lower = token.lower()
        if lower.endswith("ed") or lower.endswith("ing"):
            return index
    return min(1, len(tokens) - 1)


@dataclass(slots=True)
class InferenceOutput:
    """Single inference result."""

    answer_text: str
    confidence: float
    predicted_role: str


class InferenceEngine:
    """Runtime wrapper around a trained PropQA-Net checkpoint."""

    def __init__(self, config: ProjectConfig) -> None:
        """Load the best checkpoint for inference."""

        self.config = config
        self.device = torch.device(config.runtime.device)
        self.model, self.checkpoint = load_trained_model(config, self.device)
        self.vocabularies = self.checkpoint["vocabularies"]
        self.label_id_to_token = self.vocabularies["label_vocab"]["id_to_token"]

    def _encode_tokens(self, tokens: Sequence[str], vocab: Dict[str, Any]) -> List[int]:
        """Encode tokens with a serialized vocabulary."""

        token_to_id = vocab["token_to_id"]
        unknown_id = token_to_id.get("<unk>", 0)
        return [token_to_id.get(token.lower(), unknown_id) for token in tokens]

    def infer(self, context: str, question: str) -> InferenceOutput:
        """Run question answering for a raw context/question pair."""

        context_tokens = simple_word_tokenize(context)
        question_tokens = simple_word_tokenize(question)
        pos_tags = heuristic_pos_tags(context_tokens)
        predicate_index = infer_predicate_index(context_tokens, question)
        predicate_flags = [1 if index == predicate_index else 0 for index in range(len(context_tokens))]

        token_vocab = self.vocabularies["token_vocab"]
        pos_vocab = self.vocabularies["pos_vocab"]
        label_vocab = self.vocabularies["label_vocab"]

        context_ids = torch.tensor(
            [self._encode_tokens(context_tokens, token_vocab)],
            dtype=torch.long,
            device=self.device,
        )
        question_ids = torch.tensor(
            [self._encode_tokens(question_tokens, token_vocab)],
            dtype=torch.long,
            device=self.device,
        )
        pos_ids = torch.tensor(
            [[pos_vocab["token_to_id"].get(tag, pos_vocab["token_to_id"].get("<unk>", 0)) for tag in pos_tags]],
            dtype=torch.long,
            device=self.device,
        )
        predicate_tensor = torch.tensor([predicate_flags], dtype=torch.long, device=self.device)
        context_mask = torch.ones_like(context_ids, dtype=torch.bool)
        question_mask = torch.ones_like(question_ids, dtype=torch.bool)
        label_ids = torch.tensor(
            [[label_vocab["token_to_id"].get("O", 0)] * len(context_tokens)],
            dtype=torch.long,
            device=self.device,
        )

        batch = {
            "context_ids": context_ids,
            "pos_ids": pos_ids,
            "predicate_flags": predicate_tensor,
            "context_mask": context_mask,
            "question_ids": question_ids,
            "question_mask": question_mask,
            "label_ids": label_ids,
            "answer_starts": torch.tensor([0], dtype=torch.long, device=self.device),
            "answer_ends": torch.tensor([0], dtype=torch.long, device=self.device),
            "raw_examples": [
                {
                    "context_tokens": context_tokens,
                    "answer_tokens": [],
                }
            ],
        }

        self.model.eval()
        with torch.no_grad():
            prediction = self.model.predict(batch, self.label_id_to_token)[0]

        answer_tokens = context_tokens[prediction.start : prediction.end + 1]
        answer_text = " ".join(answer_tokens).strip() or "(no answer found)"
        return InferenceOutput(
            answer_text=answer_text,
            confidence=float(prediction.confidence),
            predicted_role=prediction.role,
        )


def _format_prediction_result(context: str, question: str, prediction: InferenceOutput) -> Dict[str, Any]:
    """Convert an inference output into a serializable result payload."""

    return {
        "context": context,
        "question": question,
        "predicted_answer": prediction.answer_text,
        "confidence": prediction.confidence,
        "predicted_role": prediction.predicted_role,
    }


def ask_question(config: ProjectConfig, context: str, question: str) -> Dict[str, Any]:
    """Run one custom question against the trained model."""

    engine = InferenceEngine(config)
    return ask_question_with_engine(engine, context, question)


def ask_question_with_engine(engine: InferenceEngine, context: str, question: str) -> Dict[str, Any]:
    """Run one custom question using an already-loaded inference engine."""

    cleaned_context = context.strip()
    cleaned_question = question.strip()
    if not cleaned_context:
        raise ValueError("Context must not be empty.")
    if not cleaned_question:
        raise ValueError("Question must not be empty.")

    prediction = engine.infer(cleaned_context, cleaned_question)
    return _format_prediction_result(cleaned_context, cleaned_question, prediction)


def run_interactive_session(config: ProjectConfig) -> None:
    """Launch a terminal loop for asking custom questions."""

    engine = InferenceEngine(config)
    print("[ask] interactive question answering session")
    print("[ask] type 'quit' at any prompt to exit")

    while True:
        context = input("\n[ask] context: ").strip()
        if context.lower() in {"quit", "exit"}:
            print("[ask] session ended")
            break
        if not context:
            print("[ask] please enter a non-empty context")
            continue

        question = input("[ask] question: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("[ask] session ended")
            break
        if not question:
            print("[ask] please enter a non-empty question")
            continue

        result = ask_question_with_engine(engine, context, question)
        print("[ask] predicted answer:", result["predicted_answer"])
        print("[ask] confidence:", f"{result['confidence']:.4f}")
        print("[ask] predicted role:", result["predicted_role"])


def demo_examples() -> List[Dict[str, str]]:
    """Return the required inference demo examples."""

    return [
        {
            "context": "The chef cooked a delicious meal in the kitchen yesterday.",
            "question": "Who cooked?",
            "expected": "The chef",
            "role": "ARG0",
        },
        {
            "context": "She sent a letter to her friend last Monday.",
            "question": "When did she send?",
            "expected": "last Monday",
            "role": "ARGM-TMP",
        },
        {
            "context": "The company announced layoffs because of budget cuts.",
            "question": "Why were layoffs announced?",
            "expected": "because of budget cuts",
            "role": "ARGM-CAU",
        },
        {
            "context": "The nurse administered the medicine to the patient after dinner.",
            "question": "What was administered?",
            "expected": "the medicine",
            "role": "ARG1",
        },
        {
            "context": "The courier delivered the package to the office at noon.",
            "question": "Where was the package delivered?",
            "expected": "to the office",
            "role": "ARGM-LOC",
        },
        {
            "context": "The engineer repaired the machine carefully with a small screwdriver.",
            "question": "How did the engineer repair the machine?",
            "expected": "carefully",
            "role": "ARGM-MNR",
        },
        {
            "context": "The board approved the proposal during the morning meeting.",
            "question": "What did the board approve?",
            "expected": "the proposal",
            "role": "ARG1",
        },
        {
            "context": "Maria gave the intern a notebook for the workshop.",
            "question": "Who received a notebook?",
            "expected": "the intern",
            "role": "ARG2",
        },
        {
            "context": "Investigators examined the site after the explosion in the warehouse.",
            "question": "Where was the explosion?",
            "expected": "in the warehouse",
            "role": "ARGM-LOC",
        },
        {
            "context": "The students presented their project enthusiastically at the science fair.",
            "question": "How did the students present their project?",
            "expected": "enthusiastically",
            "role": "ARGM-MNR",
        },
    ]


def run_demo(config: ProjectConfig) -> List[Dict[str, Any]]:
    """Run the mandatory 10-example inference demo."""

    engine = InferenceEngine(config)
    demo_results: List[Dict[str, Any]] = []

    for example in demo_examples():
        prediction = engine.infer(example["context"], example["question"])
        is_correct = normalize_text(prediction.answer_text) == normalize_text(example["expected"])
        result = {
            "context": example["context"],
            "question": example["question"],
            "predicted_answer": prediction.answer_text,
            "confidence": prediction.confidence,
            "predicted_role": prediction.predicted_role,
            "correct_role": example["role"],
            "expected_answer": example["expected"],
            "match": "CORRECT" if is_correct else "INCORRECT",
        }
        demo_results.append(result)

        print("[infer] sentence:", example["context"])
        print("[infer] question:", example["question"])
        print("[infer] predicted answer:", prediction.answer_text)
        print("[infer] confidence:", f"{prediction.confidence:.4f}")
        print("[infer] role:", prediction.predicted_role)
        print("[infer] expected role:", example["role"])
        print("[infer] match:", result["match"])

    return demo_results
