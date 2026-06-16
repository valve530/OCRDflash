from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .draft_verify import direct_accept_stats, tokenize_text, verify_draft_tokens
from .layout import should_use_native_text
from .schemas import BlockRecognition, DraftVerificationStats, LayoutBlock, NativeTextCandidate


DEFAULT_DOCUMENT_PROMPT = "Convert this document image region to Markdown."


@dataclass(slots=True)
class GenerationOptions:
    chunk_size: int = 16
    max_tokens: int = 256
    temperature: float = 0.0
    sampling: bool = False
    verify_native_text: bool = False
    enable_vlm: bool = False
    fallback_to_native: bool = True
    prompt: str = DEFAULT_DOCUMENT_PROMPT


class BlockGenerator:
    def recognize(
        self,
        image_path: Path,
        block: LayoutBlock,
        native_candidate: NativeTextCandidate | None,
        options: GenerationOptions,
    ) -> BlockRecognition | None:
        raise NotImplementedError


class TextContinuationModel:
    def generate_after_prefix(
        self,
        image_path: Path,
        block: LayoutBlock,
        prompt: str,
        accepted_prefix: str,
        options: GenerationOptions,
    ) -> str:
        raise NotImplementedError


class NativeDraftGenerator(BlockGenerator):
    def __init__(self, tokenizer: object | None = None):
        self.tokenizer = tokenizer

    def recognize(
        self,
        image_path: Path,
        block: LayoutBlock,
        native_candidate: NativeTextCandidate | None,
        options: GenerationOptions,
    ) -> BlockRecognition | None:
        _ = image_path
        if native_candidate is None:
            return None
        if not should_use_native_text(block.class_name):
            return None

        started = time.perf_counter()
        text = native_candidate.text.strip()
        if not text:
            return None

        if native_candidate.quality.direct_accept and not options.verify_native_text:
            stats = direct_accept_stats(text, self.tokenizer, options.chunk_size)
            return BlockRecognition(
                backend="pdf-native-text",
                text=text,
                tokens=stats.accepted_tokens,
                ms=(time.perf_counter() - started) * 1000.0,
                draft=stats,
            )

        if options.fallback_to_native:
            token_count = len(tokenize_text(self.tokenizer, text))
            stats = direct_accept_stats(text, self.tokenizer, options.chunk_size)
            stats.mode = "native_fallback_without_vlm"
            stats.accepted = False
            stats.prefix_accepted = False
            stats.accepted_tokens = 0
            return BlockRecognition(
                backend="pdf-native-text:fallback",
                text=text,
                tokens=token_count,
                ms=(time.perf_counter() - started) * 1000.0,
                draft=stats,
            )
        return None


class DraftVerifyingGenerator(BlockGenerator):
    def __init__(
        self,
        verifier_model: object,
        tokenizer: object,
        continuation_model: TextContinuationModel | None = None,
        prompt: str = "Convert this document image region to Markdown.",
    ):
        self.verifier_model = verifier_model
        self.tokenizer = tokenizer
        self.continuation_model = continuation_model
        self.prompt = prompt

    def recognize(
        self,
        image_path: Path,
        block: LayoutBlock,
        native_candidate: NativeTextCandidate | None,
        options: GenerationOptions,
    ) -> BlockRecognition | None:
        if native_candidate is None or not should_use_native_text(block.class_name):
            return None

        started = time.perf_counter()
        draft = native_candidate.text.strip()
        if not draft:
            return None

        stats = verify_draft_tokens(
            self.verifier_model,
            self.tokenizer,
            prompt=self.prompt,
            draft_text=draft,
            chunk_size=options.chunk_size,
        )
        if stats.accepted:
            text = draft
            backend = "draft-verified:accepted"
        elif stats.prefix_accepted and self.continuation_model is not None:
            accepted_prefix = _prefix_by_tokens(draft, stats.accepted_tokens)
            suffix = self.continuation_model.generate_after_prefix(
                image_path,
                block,
                self.prompt,
                accepted_prefix,
                options,
            )
            text = accepted_prefix + suffix
            stats.generated_tokens = len(tokenize_text(self.tokenizer, suffix))
            backend = "draft-verified:prefix"
        elif options.fallback_to_native:
            text = draft
            backend = "draft-verified:fallback-native"
        else:
            return None

        return BlockRecognition(
            backend=backend,
            text=text,
            tokens=len(tokenize_text(self.tokenizer, text)),
            ms=(time.perf_counter() - started) * 1000.0,
            draft=stats,
        )


def _prefix_by_tokens(text: str, token_count: int) -> str:
    if token_count <= 0:
        return ""
    parts = text.split()
    if len(parts) >= token_count:
        return " ".join(parts[:token_count]) + (" " if token_count < len(parts) else "")
    return text


class TransformersVlmGenerator(BlockGenerator):
    """Thin adapter placeholder for document VLMs.

    The interface is intentionally block-level so PaddleOCR-VL or any future
    HF-compatible document VLM can replace the native fallback without changing
    layout/native-text/reporting code.
    """

    def __init__(self, model: object, processor: object, tokenizer: object | None = None):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    def recognize(
        self,
        image_path: Path,
        block: LayoutBlock,
        native_candidate: NativeTextCandidate | None,
        options: GenerationOptions,
    ) -> BlockRecognition | None:
        from PIL import Image
        import torch

        started = time.perf_counter()
        prompt = options.prompt
        if native_candidate and native_candidate.text.strip():
            prompt = (
                f"{prompt}\n\nPDF native text draft:\n{native_candidate.text.strip()}\n\n"
                "Verify or correct the draft using the image."
            )

        with Image.open(image_path) as image:
            crop = image.crop((block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)).convert("RGB")
            inputs = _processor_inputs(self.processor, crop, prompt)
        inputs = _move_tensors(inputs, _model_device(self.model))

        generate_kwargs = {
            "max_new_tokens": options.max_tokens,
            "do_sample": options.sampling,
        }
        if options.sampling:
            generate_kwargs["temperature"] = options.temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)  # type: ignore[attr-defined]

        prompt_len = inputs.get("input_ids").shape[-1] if "input_ids" in inputs else 0
        generated_ids = output_ids[:, prompt_len:] if prompt_len and output_ids.ndim == 2 else output_ids
        text = _decode_generated(self.processor, generated_ids, output_ids)
        token_count = int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else 0
        return BlockRecognition(
            backend="transformers-vlm",
            text=text,
            tokens=token_count,
            ms=(time.perf_counter() - started) * 1000.0,
            draft=None,
        )


class PaddleOCRVLDFlashGenerator(BlockGenerator):
    def __init__(self, model: object, processor: object):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def recognize(
        self,
        image_path: Path,
        block: LayoutBlock,
        native_candidate: NativeTextCandidate | None,
        options: GenerationOptions,
    ) -> BlockRecognition | None:
        from PIL import Image
        import torch

        started = time.perf_counter()
        with Image.open(image_path) as image:
            crop = image.crop((block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)).convert("RGB")
            inputs = _paddleocr_vl_inputs(self.processor, crop, _paddleocr_vl_prompt(block, options.prompt))
        inputs = _move_tensors(dict(inputs), _model_device(self.model))

        draft = native_candidate.text.strip() if native_candidate and native_candidate.text.strip() else ""
        if draft and should_use_native_text(block.class_name):
            stats, generated_ids = self._generate_with_dflash(inputs, draft, options)
        else:
            stats = None
            output_ids = self._generate(inputs, options)
            generated_ids = _trim_prompt_ids(output_ids, inputs)

        text = _decode_generated(self.processor, generated_ids, generated_ids)
        tokens = int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else len(tokenize_text(self.tokenizer, text))
        backend = "paddleocr-vl:generate"
        if stats is not None:
            if stats.accepted:
                backend = "paddleocr-vl:dflash:accepted"
                text = draft
                tokens = stats.accepted_tokens
            elif stats.prefix_accepted:
                backend = "paddleocr-vl:dflash:prefix"
                tokens = stats.accepted_tokens + stats.generated_tokens
            else:
                backend = "paddleocr-vl:dflash:fallback-generate"

        return BlockRecognition(
            backend=backend,
            text=text,
            tokens=tokens,
            ms=(time.perf_counter() - started) * 1000.0,
            draft=stats,
        )

    def _generate_with_dflash(
        self,
        inputs: dict[str, object],
        draft: str,
        options: GenerationOptions,
    ):
        import torch

        draft_ids = _tokenize_draft(self.tokenizer, draft)
        if not draft_ids:
            return None, self._generate(inputs, options)

        base_input_ids = inputs["input_ids"]
        accepted = 0
        checked = 0
        chunk_size = max(1, options.chunk_size)
        with torch.no_grad():
            while accepted < len(draft_ids):
                for expected in draft_ids[accepted : accepted + chunk_size]:
                    predicted = self._predict_next_token(inputs, draft_ids[:accepted])
                    checked += 1
                    if predicted != expected:
                        generated_ids = self._continue_from_prefix(inputs, draft_ids[:accepted], options)
                        generated_count = int(generated_ids.shape[-1]) - int(base_input_ids.shape[-1]) - accepted
                        stats = DraftVerificationStats(
                            mode="paddleocr_vl_token_verify",
                            accepted=False,
                            prefix_accepted=accepted > 0,
                            draft_tokens=len(draft_ids),
                            checked_tokens=checked,
                            matched_tokens=accepted,
                            accepted_tokens=accepted,
                            rejected_tokens=len(draft_ids) - accepted,
                            rollback_tokens=1,
                            generated_tokens=max(0, generated_count),
                            chunk_size=chunk_size,
                        )
                        return stats, _trim_prompt_ids(generated_ids, inputs)
                    accepted += 1

        stats = DraftVerificationStats(
            mode="paddleocr_vl_token_verify",
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
        return stats, _tensor_from_ids(draft_ids, base_input_ids)

    def _predict_next_token(self, inputs: dict[str, object], prefix_ids: list[int]) -> int:
        import torch

        forward_inputs = _append_token_ids(inputs, prefix_ids)
        outputs = self.model(**forward_inputs, logits_to_keep=1, use_cache=False)  # type: ignore[operator]
        return int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())

    def _continue_from_prefix(
        self,
        inputs: dict[str, object],
        prefix_ids: list[int],
        options: GenerationOptions,
    ):
        generate_inputs = _append_token_ids(inputs, prefix_ids)
        return self._generate(generate_inputs, options)

    def _generate(self, inputs: dict[str, object], options: GenerationOptions):
        import torch

        kwargs = {
            "max_new_tokens": options.max_tokens,
            "do_sample": options.sampling,
        }
        if options.sampling:
            kwargs["temperature"] = options.temperature
        with torch.no_grad():
            return self.model.generate(**inputs, **kwargs)  # type: ignore[attr-defined]


def _paddleocr_vl_prompt(block: LayoutBlock, fallback_prompt: str) -> str:
    if fallback_prompt and fallback_prompt != DEFAULT_DOCUMENT_PROMPT:
        return fallback_prompt
    if block.class_name == "table":
        return "Table Recognition:"
    if block.class_name == "chart":
        return "Chart Recognition:"
    if block.class_name in {"display_formula", "inline_formula", "formula_number"}:
        return "Formula Recognition:"
    if block.class_name == "seal":
        return "Seal Recognition:"
    return "OCR:"


def _paddleocr_vl_inputs(processor: object, image: object, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if apply_chat_template is not None:
        inputs = apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return _ensure_mm_token_type_ids(processor, inputs)
    image_token = getattr(processor, "image_token", "<|IMAGE_PLACEHOLDER|>")
    inputs = processor(images=image, text=f"{image_token}{prompt}", return_tensors="pt")  # type: ignore[operator]
    return _ensure_mm_token_type_ids(processor, inputs)


def _ensure_mm_token_type_ids(processor: object, inputs: object) -> object:
    if "mm_token_type_ids" in inputs or "input_ids" not in inputs:
        return inputs
    image_token_ids = _processor_image_token_ids(processor)
    try:
        import torch

        input_ids = inputs["input_ids"]
        if image_token_ids:
            mm = torch.zeros_like(input_ids)
            for token_id in image_token_ids:
                mm = torch.where(input_ids == token_id, torch.ones_like(mm), mm)
            inputs["mm_token_type_ids"] = mm
            return inputs
        create = getattr(processor, "create_mm_token_type_ids", None)
        if create is None:
            return inputs
        mm_token_type_ids = create(input_ids.tolist())
        inputs["mm_token_type_ids"] = torch.tensor(mm_token_type_ids, dtype=input_ids.dtype, device=input_ids.device)
    except Exception:
        create = getattr(processor, "create_mm_token_type_ids", None)
        if create is not None:
            inputs["mm_token_type_ids"] = create(inputs["input_ids"].tolist())
    return inputs


def _processor_image_token_ids(processor: object) -> list[int]:
    ids: list[int] = []
    for owner in (processor, getattr(processor, "tokenizer", None)):
        if owner is None:
            continue
        value = getattr(owner, "image_token_id", None)
        if isinstance(value, int):
            ids.append(value)
        values = getattr(owner, "image_token_ids", None)
        if isinstance(values, list):
            ids.extend(item for item in values if isinstance(item, int))
    return sorted(set(ids))


def _tokenize_draft(tokenizer: object, draft: str) -> list[int]:
    encoded = tokenizer.encode(draft, add_special_tokens=False)  # type: ignore[attr-defined]
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return list(encoded)


def _append_token_ids(inputs: dict[str, object], token_ids: list[int]) -> dict[str, object]:
    import torch

    if not token_ids:
        return dict(inputs)
    input_ids = inputs["input_ids"]
    suffix = torch.tensor([token_ids], dtype=input_ids.dtype, device=input_ids.device)
    out = dict(inputs)
    out["input_ids"] = torch.cat([input_ids, suffix], dim=-1)
    if "attention_mask" in out:
        mask = out["attention_mask"]
        out["attention_mask"] = torch.cat([mask, torch.ones_like(suffix, dtype=mask.dtype)], dim=-1)
    if "mm_token_type_ids" in out:
        mm = out["mm_token_type_ids"]
        out["mm_token_type_ids"] = torch.cat([mm, torch.zeros_like(suffix, dtype=mm.dtype)], dim=-1)
    return out


def _tensor_from_ids(token_ids: list[int], like: object):
    import torch

    return torch.tensor([token_ids], dtype=like.dtype, device=like.device)


def _trim_prompt_ids(output_ids: object, inputs: dict[str, object]) -> object:
    prompt_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    if prompt_len and hasattr(output_ids, "ndim") and output_ids.ndim == 2:
        return output_ids[:, prompt_len:]
    return output_ids


def _processor_inputs(processor: object, image: object, prompt: str) -> dict[str, object]:
    try:
        return processor(images=image, text=prompt, return_tensors="pt")  # type: ignore[operator]
    except TypeError:
        return processor(text=prompt, images=image, return_tensors="pt")  # type: ignore[operator]


def _move_tensors(inputs: dict[str, object], device: str) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _model_device(model: object) -> str:
    try:
        return str(next(model.parameters()).device)  # type: ignore[attr-defined]
    except Exception:
        return "cpu"


def _decode_generated(processor: object, generated_ids: object, output_ids: object) -> str:
    decoder = getattr(processor, "batch_decode", None)
    ids = generated_ids
    if decoder is None:
        tokenizer = getattr(processor, "tokenizer", None)
        decoder = getattr(tokenizer, "batch_decode", None)
    if decoder is None:
        return str(output_ids)
    text = decoder(ids, skip_special_tokens=True)
    if isinstance(text, list):
        return text[0].strip() if text else ""
    return str(text).strip()
