"""Query-decomposition prompt/parser using the shared small chat model."""

from __future__ import annotations

import json
import re

from .chat_model import ChatModel


def _clean_question(text: str) -> str:
    cleaned = text.strip(" \n\t\r.")
    if not cleaned:
        return ""

    cleaned = cleaned[0].upper() + cleaned[1:]
    if not cleaned.endswith("?"):
        cleaned += "?"

    return cleaned


def _rule_based_fallback(
    user_query: str,
    *,
    max_questions: int,
) -> list[str]:
    parts = re.split(r"\?+", user_query.strip())
    questions: list[str] = []

    for part in parts:
        for sub in re.split(
            r"\b(?:and|also|then)\b",
            part,
            flags=re.IGNORECASE,
        ):
            question = _clean_question(sub)
            if question:
                questions.append(question)

    return _dedupe(questions)[:max_questions]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)

    return output


class QueryDecomposer:
    """Logical decomposer that reuses the already-loaded small chat model."""

    def __init__(
        self,
        *,
        model: ChatModel,
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.max_new_tokens = int(max_new_tokens)

    def decompose(
        self,
        user_query: str,
        *,
        max_questions: int = 5,
    ) -> list[str]:
        cleaned_query = str(user_query or "").strip()

        if not cleaned_query:
            return []
        if max_questions <= 0:
            raise ValueError("max_questions must be positive.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query decomposition module for a RAG system. "
                    "First clean the user's question, correct obvious typos, "
                    "and remove noisy characters. Split the user's request into "
                    "independent retrieval questions. Resolve pronouns and vague "
                    "references to their specific noun or entity when the "
                    "reference is clear from the question. Do not answer the "
                    "questions. Do not add requests that are not implied. "
                    "Return valid JSON only: a JSON array of strings. "
                    f"Return at most {max_questions} questions."
                ),
            },
            {
                "role": "user",
                "content": cleaned_query,
            },
        ]

        generation = self.model.generate(
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
        )

        questions = self._parse(generation.text)

        if not questions:
            questions = _rule_based_fallback(
                cleaned_query,
                max_questions=max_questions,
            )

        return questions[:max_questions]

    @staticmethod
    def _parse(raw_output: str) -> list[str]:
        raw_output = raw_output.strip()

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = None

        if parsed is None:
            match = re.search(
                r"\[[\s\S]*\]",
                raw_output,
            )
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None

        if not isinstance(parsed, list):
            return []

        questions = [
            _clean_question(item)
            for item in parsed
            if isinstance(item, str)
        ]

        return _dedupe(
            [question for question in questions if question]
        )
