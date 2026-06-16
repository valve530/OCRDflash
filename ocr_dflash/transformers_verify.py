from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TransformersNextTokenVerifier:
    """Next-token verifier for Hugging Face autoregressive models.

    This adapter intentionally keeps the contract small: given already-tokenized
    context ids, return the greedy next token. Vision-language models can wrap
    this idea with their own processor/visual-prefix handling while still using
    `DraftVerifyingGenerator`.
    """

    model: object
    device: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            try:
                self.device = str(next(self.model.parameters()).device)  # type: ignore[attr-defined]
            except Exception:
                self.device = "cpu"

    def next_token_id(self, context_ids: list[int]) -> int:
        if not context_ids:
            raise ValueError("context_ids must not be empty")
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)  # type: ignore[operator]
            logits = outputs.logits[:, -1, :]
            return int(torch.argmax(logits, dim=-1).item())
