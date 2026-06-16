from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schemas import DraftVerificationStats


class SimpleTokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class NextTokenModel(Protocol):
    def next_token_id(self, context_ids: list[int]) -> int: ...


@dataclass(slots=True)
class WhitespaceTokenizer:
    """Fallback tokenizer for tests and native-text-only baselines."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        pieces = text.split()
        if not pieces and text:
            pieces = [text]
        return [stable_token_id(piece) for piece in pieces]


def stable_token_id(piece: str) -> int:
    value = 2166136261
    for byte in piece.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def tokenize_text(tokenizer: object | None, text: str) -> list[int]:
    if tokenizer is None:
        return WhitespaceTokenizer().encode(text)
    encoded = tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    if isinstance(encoded, dict) and "input_ids" in encoded:
        ids = encoded["input_ids"]
        return list(ids[0] if ids and isinstance(ids[0], list) else ids)
    return list(encoded)


def direct_accept_stats(text: str, tokenizer: object | None, chunk_size: int) -> DraftVerificationStats:
    draft_tokens = len(tokenize_text(tokenizer, text))
    return DraftVerificationStats(
        mode="direct_accept",
        accepted=True,
        prefix_accepted=True,
        draft_tokens=draft_tokens,
        checked_tokens=draft_tokens,
        matched_tokens=draft_tokens,
        accepted_tokens=draft_tokens,
        rejected_tokens=0,
        rollback_tokens=0,
        generated_tokens=0,
        chunk_size=max(1, chunk_size),
    )


def verify_draft_tokens(
    model: NextTokenModel,
    tokenizer: object,
    prompt: str,
    draft_text: str,
    chunk_size: int = 16,
) -> DraftVerificationStats:
    prompt_ids = tokenize_text(tokenizer, prompt)
    draft_ids = tokenize_text(tokenizer, draft_text)
    accepted = 0
    checked = 0
    chunk_size = max(1, chunk_size)

    while accepted < len(draft_ids):
        chunk = draft_ids[accepted : accepted + chunk_size]
        for token_id in chunk:
            predicted = model.next_token_id(prompt_ids + draft_ids[:accepted])
            checked += 1
            if predicted != token_id:
                rejected = len(draft_ids) - accepted
                return DraftVerificationStats(
                    mode="token_verify",
                    accepted=False,
                    prefix_accepted=accepted > 0,
                    draft_tokens=len(draft_ids),
                    checked_tokens=checked,
                    matched_tokens=accepted,
                    accepted_tokens=accepted,
                    rejected_tokens=rejected,
                    rollback_tokens=1,
                    generated_tokens=0,
                    chunk_size=chunk_size,
                )
            accepted += 1

    return DraftVerificationStats(
        mode="token_verify",
        accepted=True,
        prefix_accepted=accepted > 0,
        draft_tokens=len(draft_ids),
        checked_tokens=checked,
        matched_tokens=accepted,
        accepted_tokens=accepted,
        rejected_tokens=0,
        rollback_tokens=0,
        generated_tokens=0,
        chunk_size=chunk_size,
    )
