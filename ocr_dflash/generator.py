from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .draft_verify import direct_accept_stats, tokenize_text, verify_draft_tokens
from .layout import should_use_native_text
from .schemas import BlockRecognition, DraftVerificationStats, LayoutBlock, NativeTextCandidate


DEFAULT_DOCUMENT_PROMPT = "Convert this document image region to Markdown."


@dataclass(slots=True)
class GenerationOptions:
    chunk_size: int = 16
    batch_size: int = 32
    batch_max_pixels: int = 0
    max_tokens: int = 256
    temperature: float = 0.0
    sampling: bool = False
    verify_native_text: bool = False
    enable_vlm: bool = False
    fallback_to_native: bool = True
    prompt: str = DEFAULT_DOCUMENT_PROMPT


@dataclass(slots=True)
class _PreparedPaddleBatch:
    group: list[tuple[int, Path, LayoutBlock, NativeTextCandidate | None]]
    inputs: object


@dataclass(slots=True)
class _PreparedPaddleRecognition:
    index: int
    image_path: Path
    block: LayoutBlock
    native_candidate: NativeTextCandidate | None
    inputs: object
    draft: str
    started: float


class BlockGenerator:
    def recognize(
        self,
        image_path: Path,
        block: LayoutBlock,
        native_candidate: NativeTextCandidate | None,
        options: GenerationOptions,
    ) -> BlockRecognition | None:
        raise NotImplementedError

    def recognize_many(
        self,
        image_path: Path,
        requests: list[tuple[LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return [self.recognize(image_path, block, native_candidate, options) for block, native_candidate in requests]

    def recognize_many_with_paths(
        self,
        requests: list[tuple[Path, LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return [self.recognize(image_path, block, native_candidate, options) for image_path, block, native_candidate in requests]


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

    def recognize_many(
        self,
        image_path: Path,
        requests: list[tuple[LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return [self.recognize(image_path, block, native_candidate, options) for block, native_candidate in requests]

    def recognize_many_with_paths(
        self,
        requests: list[tuple[Path, LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return [self.recognize(image_path, block, native_candidate, options) for image_path, block, native_candidate in requests]


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

    def recognize_many(
        self,
        image_path: Path,
        requests: list[tuple[LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return [self.recognize(image_path, block, native_candidate, options) for block, native_candidate in requests]

    def recognize_many_with_paths(
        self,
        requests: list[tuple[Path, LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return [self.recognize(image_path, block, native_candidate, options) for image_path, block, native_candidate in requests]


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

    def recognize_many(
        self,
        image_path: Path,
        requests: list[tuple[LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return [self.recognize(image_path, block, native_candidate, options) for block, native_candidate in requests]

    def recognize_many_with_paths(
        self,
        requests: list[tuple[Path, LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return [self.recognize(image_path, block, native_candidate, options) for image_path, block, native_candidate in requests]


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
        return self.recognize_with_image(None, image_path, block, native_candidate, options)

    def recognize_with_image(
        self,
        image: object | None,
        image_path: Path,
        block: LayoutBlock,
        native_candidate: NativeTextCandidate | None,
        options: GenerationOptions,
    ) -> BlockRecognition | None:
        started = time.perf_counter()
        if image is None:
            from PIL import Image

            with Image.open(image_path) as raw_image:
                return self._recognize_from_image(raw_image.convert("RGB"), block, native_candidate, options, started)
        return self._recognize_from_image(image, block, native_candidate, options, started)

    def _recognize_from_image(
        self,
        image: object,
        block: LayoutBlock,
        native_candidate: NativeTextCandidate | None,
        options: GenerationOptions,
        started: float,
    ) -> BlockRecognition | None:
        crop = image.crop((block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)).convert("RGB")  # type: ignore[attr-defined]
        inputs = _paddleocr_vl_inputs(self.processor, crop, _paddleocr_vl_prompt(block, options.prompt))
        inputs = _move_tensors(dict(inputs), _model_device(self.model))

        draft = _normalize_dflash_draft(native_candidate.text).strip() if native_candidate and native_candidate.text.strip() else ""
        if draft and should_use_native_text(block.class_name):
            stats, generated_ids = self._generate_with_dflash(inputs, draft, options)
        else:
            stats = None
            output_ids = self._generate(inputs, options)
            generated_ids = _trim_prompt_ids(output_ids, inputs)

        backend = "paddleocr-vl:generate"
        text: str
        tokens: int
        if stats is not None:
            if stats.accepted:
                backend = "paddleocr-vl:dflash:accepted"
                text = draft
                tokens = stats.accepted_tokens
            elif stats.prefix_accepted:
                backend = "paddleocr-vl:dflash:prefix"
                text = _decode_generated(self.processor, generated_ids, generated_ids)
                tokens = stats.accepted_tokens + stats.generated_tokens
            else:
                backend = "paddleocr-vl:dflash:fallback-generate"
                text = _decode_generated(self.processor, generated_ids, generated_ids)
                tokens = int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else len(tokenize_text(self.tokenizer, text))
        else:
            text = _decode_generated(self.processor, generated_ids, generated_ids)
            tokens = int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else len(tokenize_text(self.tokenizer, text))

        return BlockRecognition(
            backend=backend,
            text=text,
            tokens=tokens,
            ms=(time.perf_counter() - started) * 1000.0,
            draft=stats,
        )

    def recognize_many(
        self,
        image_path: Path,
        requests: list[tuple[LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return self.recognize_many_with_paths([(image_path, block, native_candidate) for block, native_candidate in requests], options)

    def recognize_many_with_paths(
        self,
        requests: list[tuple[Path, LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        if not requests:
            return []

        opened_images: dict[Path, object] = {}
        from PIL import Image

        def _get_image(path: Path) -> object:
            image = opened_images.get(path)
            if image is not None:
                return image
            with Image.open(path) as raw_image:
                image = raw_image.convert("RGB")
            opened_images[path] = image
            return image

        for image_path, _block, _native_candidate in requests:
            _get_image(image_path)

        direct: list[tuple[int, Path, LayoutBlock, NativeTextCandidate | None]] = []
        batchable: list[tuple[int, Path, LayoutBlock, NativeTextCandidate | None]] = []
        for index, (image_path, block, native_candidate) in enumerate(requests):
            draft = _normalize_dflash_draft(native_candidate.text).strip() if native_candidate and native_candidate.text.strip() else ""
            if draft and should_use_native_text(block.class_name):
                direct.append((index, image_path, block, native_candidate))
            else:
                batchable.append((index, image_path, block, native_candidate))

        results: list[BlockRecognition | None] = [None] * len(requests)
        if direct:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _prepare_paddleocr_vl_recognition_from_cache,
                    self.processor,
                    direct[0],
                    options.prompt,
                    opened_images,
                )
                for direct_index, _request in enumerate(direct):
                    prepared = future.result()
                    if direct_index + 1 < len(direct):
                        future = executor.submit(
                            _prepare_paddleocr_vl_recognition_from_cache,
                            self.processor,
                            direct[direct_index + 1],
                            options.prompt,
                            opened_images,
                        )
                    results[prepared.index] = self._recognize_prepared(prepared, options)

        if batchable:
            batch_size = max(1, int(getattr(options, "batch_size", 8)))
            max_tile_pixels = int(getattr(options, "batch_max_pixels", 0) or 0)
            batchable.sort(key=_paddleocr_vl_batch_sort_key, reverse=True)
            batch_groups: list[list[tuple[int, Path, LayoutBlock, NativeTextCandidate | None]]] = []
            for start in range(0, len(batchable), batch_size):
                group = batchable[start : start + batch_size]
                if max_tile_pixels > 0:
                    group = _split_batch_by_pixels(group, max_tile_pixels)
                batch_groups.append(group)

            if not batch_groups:
                return results

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _prepare_paddleocr_vl_batch_from_cache,
                    self.processor,
                    batch_groups[0],
                    options.prompt,
                    opened_images,
                )
                for batch_index, group in enumerate(batch_groups):
                    started = time.perf_counter()
                    prepared = future.result()
                    if batch_index + 1 < len(batch_groups):
                        future = executor.submit(
                            _prepare_paddleocr_vl_batch_from_cache,
                            self.processor,
                            batch_groups[batch_index + 1],
                            options.prompt,
                            opened_images,
                        )

                    inputs = _move_tensors(dict(prepared.inputs), _model_device(self.model))
                    _ = prepared.group
                    output_ids = self._generate(inputs, options)
                    generated_ids = _trim_prompt_ids(output_ids, inputs)
                    texts = _decode_generated_batch(self.processor, generated_ids, output_ids)
                    if len(texts) != len(group):
                        texts = [texts[0] if texts else ""] * len(group)
                    for (index, _image_path, _block, _native_candidate), text in zip(group, texts):
                        tokens = int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else len(tokenize_text(self.tokenizer, text))
                        results[index] = BlockRecognition(
                            backend="paddleocr-vl:generate",
                            text=text,
                            tokens=tokens,
                            ms=(time.perf_counter() - started) * 1000.0,
                            draft=None,
                        )

        for image in opened_images.values():
            try:
                image.close()
            except Exception:
                pass

        return results

    def recognize_many_with_image(
        self,
        image_path: Path,
        image: object,
        requests: list[tuple[LayoutBlock, NativeTextCandidate | None]],
        options: GenerationOptions,
    ) -> list[BlockRecognition | None]:
        return [
            self._recognize_from_image(image, block, native_candidate, options, time.perf_counter())
            for block, native_candidate in requests
        ]

    def _recognize_prepared(
        self,
        prepared: _PreparedPaddleRecognition,
        options: GenerationOptions,
    ) -> BlockRecognition | None:
        return self._run_recognition(
            prepared.inputs,
            prepared.block,
            prepared.native_candidate,
            options,
            prepared.started,
            prepared.draft,
        )

    def _run_recognition(
        self,
        inputs: dict[str, object],
        block: LayoutBlock,
        native_candidate: NativeTextCandidate | None,
        options: GenerationOptions,
        started: float,
        draft: str | None = None,
    ) -> BlockRecognition | None:
        draft_text = _normalize_dflash_draft(native_candidate.text).strip() if draft is None and native_candidate and native_candidate.text.strip() else (draft or "")
        if draft_text and should_use_native_text(block.class_name):
            stats, generated_ids = self._generate_with_dflash(inputs, draft_text, options)
        else:
            stats = None
            output_ids = self._generate(inputs, options)
            generated_ids = _trim_prompt_ids(output_ids, inputs)

        backend = "paddleocr-vl:generate"
        text: str
        tokens: int
        if stats is not None:
            if stats.accepted:
                backend = "paddleocr-vl:dflash:accepted"
                text = draft_text
                tokens = stats.accepted_tokens
            elif stats.prefix_accepted:
                backend = "paddleocr-vl:dflash:prefix"
                text = _decode_generated(self.processor, generated_ids, generated_ids)
                tokens = stats.accepted_tokens + stats.generated_tokens
            else:
                backend = "paddleocr-vl:dflash:fallback-generate"
                text = _decode_generated(self.processor, generated_ids, generated_ids)
                tokens = int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else len(tokenize_text(self.tokenizer, text))
        else:
            text = _decode_generated(self.processor, generated_ids, generated_ids)
            tokens = int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else len(tokenize_text(self.tokenizer, text))

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
        draft_ids = _tokenize_draft(self.tokenizer, draft)
        if not draft_ids:
            return None, self._generate(inputs, options)
        chunks = _split_dflash_chunks(self.tokenizer, draft_ids, options.chunk_size)

        try:
            return self._generate_with_dflash_cache(inputs, draft_ids, chunks, options)
        except (AttributeError, TypeError, RuntimeError) as exc:
            if not _should_fallback_from_cache_error(exc):
                raise
            return self._generate_with_dflash_recompute(inputs, draft_ids, chunks, options)

    def _generate_with_dflash_cache(
        self,
        inputs: dict[str, object],
        draft_ids: list[int],
        chunks: list[list[int]],
        options: GenerationOptions,
    ):
        import torch

        base_input_ids = inputs["input_ids"]
        accepted = 0
        checked = 0
        chunk_size = max(1, options.chunk_size)
        generated_ids: list[int] = []
        with torch.no_grad():
            logits, cache = self._prefill_cache(inputs)
            base_cache_len = _cache_seq_length(cache)
            for chunk in chunks:
                chunk_outputs = self._forward_token_ids(chunk, cache, base_input_ids)
                predicted = _chunk_predictions(logits, chunk_outputs.logits)
                chunk_matches = 0
                for actual, expected in zip(predicted, chunk):
                    checked += 1
                    if actual != expected:
                        break
                    chunk_matches += 1

                accepted += chunk_matches
                if chunk_matches == len(chunk):
                    logits = chunk_outputs.logits
                    cache = chunk_outputs.past_key_values
                    continue

                rollback_to = base_cache_len + accepted
                _crop_cache(chunk_outputs.past_key_values, rollback_to)
                cache = chunk_outputs.past_key_values
                correction = predicted[chunk_matches]
                if not _is_eos(self.model, correction):
                    generated_ids.append(correction)
                    logits, cache = self._decode_one_with_cache(correction, cache, base_input_ids)
                generated_ids.extend(self._continue_greedy_from_cache(logits, cache, options.max_tokens, base_input_ids))

                out_ids = draft_ids[:accepted] + generated_ids
                generated_count = len(generated_ids)
                stats = DraftVerificationStats(
                    mode="paddleocr_vl_chunk_verify",
                    accepted=False,
                    prefix_accepted=accepted > 0,
                    draft_tokens=len(draft_ids),
                    checked_tokens=checked,
                    matched_tokens=accepted,
                    accepted_tokens=accepted,
                    rejected_tokens=len(draft_ids) - accepted,
                    rollback_tokens=1,
                    generated_tokens=generated_count,
                    chunk_size=chunk_size,
                )
                return stats, _tensor_from_ids(out_ids, base_input_ids)

        stats = DraftVerificationStats(
            mode="paddleocr_vl_chunk_verify",
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

    def _generate_with_dflash_recompute(
        self,
        inputs: dict[str, object],
        draft_ids: list[int],
        chunks: list[list[int]],
        options: GenerationOptions,
    ):
        import torch

        base_input_ids = inputs["input_ids"]
        accepted = 0
        checked = 0
        chunk_size = max(1, options.chunk_size)
        with torch.no_grad():
            for chunk in chunks:
                for expected in chunk:
                    predicted = self._predict_next_token_recompute(inputs, draft_ids[:accepted])
                    checked += 1
                    if predicted != expected:
                        generated_ids = self._continue_from_prefix_recompute(inputs, draft_ids[:accepted], options)
                        generated_count = int(generated_ids.shape[-1]) - int(base_input_ids.shape[-1]) - accepted
                        stats = DraftVerificationStats(
                            mode="paddleocr_vl_token_verify_recompute",
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
            mode="paddleocr_vl_token_verify_recompute",
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

    def _prefill_cache(self, inputs: dict[str, object]):
        outputs = self.model(**inputs, use_cache=True, logits_to_keep=1)  # type: ignore[operator]
        return outputs.logits, outputs.past_key_values

    def _forward_token_ids(self, token_ids: list[int], cache: object, like: object):
        input_ids = _tensor_from_ids(token_ids, like)
        return self.model(  # type: ignore[operator]
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=len(token_ids),
        )

    def _decode_one_with_cache(self, token_id: int, cache: object, like: object):
        outputs = self._forward_token_ids([token_id], cache, like)
        return outputs.logits, outputs.past_key_values

    def _continue_greedy_from_cache(
        self,
        logits: object,
        cache: object,
        max_tokens: int,
        like: object,
    ) -> list[int]:
        import torch

        generated: list[int] = []
        current_logits = logits
        current_cache = cache
        for _ in range(max(0, max_tokens)):
            token = int(torch.argmax(current_logits[:, -1, :], dim=-1).item())
            if _is_eos(self.model, token):
                break
            generated.append(token)
            current_logits, current_cache = self._decode_one_with_cache(token, current_cache, like)
        return generated

    def _predict_next_token_recompute(self, inputs: dict[str, object], prefix_ids: list[int]) -> int:
        import torch

        forward_inputs = _append_token_ids(inputs, prefix_ids)
        outputs = self.model(**forward_inputs, logits_to_keep=1, use_cache=False)  # type: ignore[operator]
        return int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())

    def _continue_from_prefix_recompute(
        self,
        inputs: dict[str, object],
        prefix_ids: list[int],
        options: GenerationOptions,
    ):
        generate_inputs = _append_token_ids(inputs, prefix_ids)
        return self._generate(generate_inputs, options)

    def _generate(self, inputs: dict[str, object], options: GenerationOptions):
        import torch

        if not options.sampling:
            try:
                return self._generate_with_cache(inputs, options.max_tokens)
            except (AttributeError, TypeError, RuntimeError) as exc:
                if not _should_fallback_from_cache_error(exc):
                    raise

        kwargs = {
            "max_new_tokens": options.max_tokens,
            "do_sample": options.sampling,
        }
        if options.sampling:
            kwargs["temperature"] = options.temperature
        with torch.no_grad():
            return self.model.generate(**inputs, **kwargs)  # type: ignore[attr-defined]

    def _generate_with_cache(self, inputs: dict[str, object], max_tokens: int):
        import torch

        with torch.no_grad():
            logits, cache = self._prefill_cache(inputs)
            generated = self._continue_greedy_from_cache(logits, cache, max_tokens, inputs["input_ids"])
        if not generated:
            return inputs["input_ids"]
        generated_ids = _tensor_from_ids(generated, inputs["input_ids"])
        return torch.cat([inputs["input_ids"], generated_ids], dim=-1)


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


def _paddleocr_vl_width_bucket(block: LayoutBlock) -> int:
    width = max(0.0, block.bbox.width)
    return int(width // 128)


def _paddleocr_vl_batch_sort_key(item: tuple[int, Path, LayoutBlock, NativeTextCandidate | None]) -> tuple[float, float, float]:
    _index, _image_path, block, _native_candidate = item
    width = max(0.0, block.bbox.width)
    height = max(0.0, block.bbox.height)
    return (height, width, block.bbox.area)


def _split_batch_by_pixels(
    group: list[tuple[int, Path, LayoutBlock, NativeTextCandidate | None]],
    max_tile_pixels: int,
) -> list[tuple[int, Path, LayoutBlock, NativeTextCandidate | None]]:
    if max_tile_pixels <= 0 or len(group) <= 1:
        return group
    total_pixels = sum(max(1, int(block.bbox.width * block.bbox.height)) for _index, _image_path, block, _native_candidate in group)
    if total_pixels <= max_tile_pixels:
        return group
    midpoint = len(group) // 2
    left = group[:midpoint]
    right = group[midpoint:]
    return _split_batch_by_pixels(left, max_tile_pixels) + _split_batch_by_pixels(right, max_tile_pixels)


def _prepare_paddleocr_vl_recognition_from_cache(
    processor: object,
    request: tuple[int, Path, LayoutBlock, NativeTextCandidate | None],
    prompt: str,
    opened_images: dict[Path, object],
) -> _PreparedPaddleRecognition:
    index, image_path, block, native_candidate = request
    image = opened_images.get(image_path)
    if image is None:
        from PIL import Image

        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        opened_images[image_path] = image
    crop = image.crop((block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)).convert("RGB")
    inputs = _paddleocr_vl_inputs(processor, crop, _paddleocr_vl_prompt(block, prompt))
    draft = _normalize_dflash_draft(native_candidate.text).strip() if native_candidate and native_candidate.text.strip() else ""
    return _PreparedPaddleRecognition(
        index=index,
        image_path=image_path,
        block=block,
        native_candidate=native_candidate,
        inputs=inputs,
        draft=draft,
        started=time.perf_counter(),
    )


def _prepare_paddleocr_vl_batch(
    processor: object,
    group: list[tuple[int, Path, LayoutBlock, NativeTextCandidate | None]],
    prompt: str,
) -> _PreparedPaddleBatch:
    from PIL import Image

    opened_images: dict[Path, object] = {}
    try:
        crops = []
        for _index, image_path, block, _native_candidate in group:
            image = opened_images.get(image_path)
            if image is None:
                with Image.open(image_path) as raw_image:
                    image = raw_image.convert("RGB")
                opened_images[image_path] = image
            crops.append(image.crop((block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)))
        prompts = [_paddleocr_vl_prompt(block, prompt) for _, _, block, _ in group]
        inputs = _paddleocr_vl_batch_inputs(processor, crops, prompts)
        return _PreparedPaddleBatch(group=group, inputs=inputs)
    finally:
        for image in opened_images.values():
            try:
                image.close()
            except Exception:
                pass


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


def _paddleocr_vl_batch_inputs(processor: object, images: list[object], prompts: list[str]):
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if apply_chat_template is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
            for image, prompt in zip(images, prompts)
        ]
        inputs = apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return _ensure_mm_token_type_ids(processor, inputs)
    image_token = getattr(processor, "image_token", "<|IMAGE_PLACEHOLDER|>")
    try:
        inputs = processor(images=images, text=[f"{image_token}{prompt}" for prompt in prompts], return_tensors="pt")  # type: ignore[operator]
    except TypeError:
        inputs = processor(images=images, text="\n".join(f"{image_token}{prompt}" for prompt in prompts), return_tensors="pt")  # type: ignore[operator]
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


def _normalize_dflash_draft(text: str) -> str:
    out: list[str] = []
    chars = list(text)
    for index, ch in enumerate(chars):
        if ch.isspace():
            prev = next((item for item in reversed(chars[:index]) if not item.isspace()), None)
            next_ = next((item for item in chars[index + 1 :] if not item.isspace()), None)
            if _should_drop_dflash_space(prev, next_):
                continue
            if not out or out[-1] not in {" ", "\n"}:
                out.append(" ")
            continue
        out.append(ch)
    return _canonicalize_vlm_draft("".join(out))


def _canonicalize_vlm_draft(text: str) -> str:
    text = (
        text.replace("∗", r"\(^{*}\)")
        .replace("†", r"\(^{\dagger}\)")
        .replace("‡", r"\(^{\ddagger}\)")
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
    )
    text = re.sub(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]\s*", lambda m: rf" \([{m.group(1).replace(' ', '')}]\) ", text)
    text = re.sub(r"(?<=[A-Za-z0-9\]\)])([,;:])(?=[A-Za-z0-9])", r"\1 ", text)
    text = re.sub(r"(?<=[A-Za-z\]\)])\.(?=[A-Z])", ". ", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\((?=[A-Za-z0-9])", " (", text)
    text = re.sub(r" {2,}", " ", text)
    return text


def _should_drop_dflash_space(prev: str | None, next_: str | None) -> bool:
    if prev is None or next_ is None:
        return True
    return (
        (_is_cjk(prev) and next_.isascii() and next_.isdigit())
        or (prev.isascii() and prev.isdigit() and _is_cjk(next_))
        or (_is_cjk(prev) and next_ in {"%", ".", ",", "/", "-"})
        or (prev in {"%", ".", ",", "/", "-"} and _is_cjk(next_))
    )


def _split_dflash_chunks(tokenizer: object, draft_ids: list[int], core_tokens_per_chunk: int) -> list[list[int]]:
    core_tokens_per_chunk = max(1, core_tokens_per_chunk)
    chunks: list[list[int]] = []
    start = 0
    while start < len(draft_ids):
        if _is_special_chunk_token(tokenizer, draft_ids[start]):
            chunks.append([draft_ids[start]])
            start += 1
            continue

        end = start
        core_count = 0
        while (
            end < len(draft_ids)
            and core_count < core_tokens_per_chunk
            and not _is_special_chunk_token(tokenizer, draft_ids[end])
        ):
            end += 1
            core_count += 1
        chunks.append(draft_ids[start:end])
        start = end
    return chunks


def _is_special_chunk_token(tokenizer: object, token_id: int) -> bool:
    piece = _decode_token_piece(tokenizer, token_id)
    return bool(piece) and all(_is_special_chunk_char(ch) for ch in piece)


def _decode_token_piece(tokenizer: object, token_id: int) -> str:
    decode = getattr(tokenizer, "decode", None)
    if decode is not None:
        try:
            return str(decode([token_id], skip_special_tokens=False))
        except TypeError:
            return str(decode([token_id]))
    batch_decode = getattr(tokenizer, "batch_decode", None)
    if batch_decode is not None:
        try:
            return str(batch_decode([[token_id]], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])
        except TypeError:
            return str(batch_decode([[token_id]])[0])
    return ""


def _is_special_chunk_char(ch: str) -> bool:
    return ch.isspace() or ch in {
        "\u00a0",
        "\n",
        "\r",
        "\t",
        ",",
        ".",
        ";",
        ":",
        "!",
        "?",
        "%",
        ")",
        "]",
        "}",
        "）",
        "】",
        "」",
        "》",
        "，",
        "。",
        "；",
        "：",
        "！",
        "？",
        "、",
        "…",
        "—",
        "-",
        "_",
    }


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0x2CEB0 <= code <= 0x2EBEF
        or 0x30000 <= code <= 0x3134F
    )


def _tokenize_draft(tokenizer: object, draft: str) -> list[int]:
    encoded = tokenizer.encode(draft, add_special_tokens=False)  # type: ignore[attr-defined]
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return list(encoded)


def _chunk_predictions(prefix_logits: object, chunk_logits: object) -> list[int]:
    import torch

    first = torch.argmax(prefix_logits[:, -1, :], dim=-1).reshape(1)
    if chunk_logits.shape[-2] <= 1:
        return [int(first.item())]
    rest = torch.argmax(chunk_logits[:, :-1, :], dim=-1).reshape(-1)
    return [int(item) for item in torch.cat([first, rest], dim=0).tolist()]


def _cache_seq_length(cache: object) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if getter is not None:
        return int(getter())
    try:
        first = cache[0][0]
        return int(first.shape[-2])
    except Exception:
        return 0


def _crop_cache(cache: object, max_length: int) -> None:
    crop = getattr(cache, "crop", None)
    if crop is None:
        raise RuntimeError("cache crop unsupported")
    crop(max_length)


def _is_eos(model: object, token_id: int) -> bool:
    config = getattr(model, "config", None)
    eos = getattr(config, "eos_token_id", None)
    if isinstance(eos, list):
        return token_id in eos
    return eos is not None and token_id == int(eos)


def _should_fallback_from_cache_error(exc: Exception) -> bool:
    message = str(exc)
    return any(
        marker in message
        for marker in (
            "past_key_values",
            "cache",
            "logits_to_keep",
            "cannot infer cache",
            "use_cache",
        )
    )


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


def _decode_generated_batch(processor: object, generated_ids: object, output_ids: object) -> list[str]:
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None:
        tokenizer = getattr(processor, "tokenizer", None)
        decoder = getattr(tokenizer, "batch_decode", None)
    if decoder is None:
        return [str(output_ids)]
    texts = decoder(generated_ids, skip_special_tokens=True)
    if isinstance(texts, list):
        return [str(text).strip() for text in texts]
    return [str(texts).strip()]
