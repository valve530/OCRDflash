from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch

from ocr_dflash.schemas import DraftVerificationStats


PROMPTS = {
    "ocr": "OCR:",
    "text": "OCR:",
    "content": "OCR:",
    "paragraph_title": "OCR:",
    "doc_title": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
    "display_formula": "Formula Recognition:",
    "inline_formula": "Formula Recognition:",
    "formula_number": "Formula Recognition:",
    "seal": "Seal Recognition:",
}

DEBUG_IMAGE = Path("tmp/simple_dflash_run4/page_0000/crops/block_0001.png")
DEBUG_MODEL = Path("models/PaddleOCR-VL-1.6")
DEBUG_TASK = "doc_title"
DEBUG_MODE = "dflash"
DEBUG_DRAFT = "Attention Is All You Need"

THE_FIRST_LIGATURE_RE = re.compile(r"\btheﬁrst\b")
BEEN_FIRMLY_LIGATURE_RE = re.compile(r"\bbeenﬁrmly\b")
CITATION_RE = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]\s*")
PUNCT_NO_SPACE_RE = re.compile(r"(?<=[A-Za-z0-9\]\)])([,;:])(?=[A-Za-z0-9])")
SENTENCE_NO_SPACE_RE = re.compile(r"(?<=[A-Za-z\]\)])\.(?=[A-Z])")
INITIAL_SPLIT_RE = re.compile(r"(?<=\b[A-Z])\.\s+(?=[A-Z][a-z])")
PAREN_NO_SPACE_RE = re.compile(r"(?<=[A-Za-z0-9])\((?=[A-Za-z0-9])")
SUPERSCRIPT_NO_SPACE_RE = re.compile(r"((?:\\\(\^\{[^}]+}\))+)(?=[A-Z])")
MULTISPACE_RE = re.compile(r" {2,}")


@dataclass(slots=True)
class VlmResult:
    text: str
    backend: str
    ms: float
    tokens: int
    draft: DraftVerificationStats | None = None


class PaddleOCRVLRunner:
    def __init__(
        self,
        model_path: str | Path = "models/PaddleOCR-VL-1.6",
        *,
        device: str = "cuda",
        dtype: str = "bf16",
        max_pixels: int = 1280 * 28 * 28,
    ):

        from transformers import AutoModelForCausalLM, AutoProcessor

        patch_transformers_for_paddleocr_vl()
        self.device = (
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        torch_dtype = (
            torch.bfloat16
            if dtype in {"bf16", "bfloat16"} and self.device == "cuda"
            else torch.float32
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                str(model_path),
                dtype=torch_dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.max_pixels = max_pixels

    def inputs(self, image: object, task: str) -> dict[str, object]:
        prompt = PROMPTS.get(task, "OCR:")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": self.processor.image_processor.min_pixels,
                    "longest_edge": self.max_pixels,
                }
            },
        )
        return inputs.to(self.model.device)

    def generate(
        self, image: object, task: str, *, max_new_tokens: int = 512
    ) -> VlmResult:
        """Baseline path, intentionally close to transformers_demo.py."""

        started = time.perf_counter()
        inputs = self.inputs(image, task)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return VlmResult(
            text=text,
            backend="vlm-generate",
            ms=(time.perf_counter() - started) * 1000.0,
            tokens=int(generated_ids.shape[-1]),
        )

    def dflash_generate(
        self,
        image: object,
        task: str,
        draft_text: str | None,
        *,
        chunk_size: int = 8,
        max_new_tokens: int = 512,
    ) -> VlmResult:
        draft = normalize_dflash_draft(draft_text or "").strip()
        if not draft:
            result = self.generate(image, task, max_new_tokens=max_new_tokens)
            result.backend = "dflash:no-draft-generate"
            return result

        started = time.perf_counter()
        inputs = self.inputs(image, task)
        draft_ids = tokenize_draft(self.tokenizer, draft)
        if not draft_ids:
            result = self.generate(image, task, max_new_tokens=max_new_tokens)
            result.backend = "dflash:empty-draft-generate"
            return result

        checked = 0
        accepted = 0
        with torch.no_grad():
            try:
                logits, cache = self._prefill(inputs)
                for chunk in split_chunks(self.tokenizer, draft_ids, chunk_size):
                    chunk_outputs = self._forward_ids(chunk, cache, inputs["input_ids"])
                    predicted = chunk_predictions(logits, chunk_outputs.logits)
                    matched = 0
                    for actual, expected in zip(predicted, chunk):
                        checked += 1
                        if actual != expected:
                            break
                        matched += 1
                    accepted += matched
                    if matched == len(chunk):
                        logits = chunk_outputs.logits
                        cache = chunk_outputs.past_key_values
                        continue

                    suffix_ids, suffix = self._generate_suffix(
                        inputs,
                        draft_ids[:accepted],
                        max_new_tokens,
                    )
                    stats = DraftVerificationStats(
                        mode="dflash_cache_verify",
                        accepted=False,
                        prefix_accepted=accepted > 0,
                        draft_tokens=len(draft_ids),
                        checked_tokens=checked,
                        matched_tokens=accepted,
                        accepted_tokens=accepted,
                        rejected_tokens=len(draft_ids) - accepted,
                        rollback_tokens=1,
                        generated_tokens=int(suffix_ids.shape[-1]),
                        chunk_size=max(1, chunk_size),
                    )
                    return VlmResult(
                        text=decode_token_ids(
                            self.tokenizer,
                            draft_ids[:accepted] + tensor_to_ids(suffix_ids),
                        ),
                        backend="dflash:prefix"
                        if accepted
                        else "dflash:fallback-generate",
                        ms=(time.perf_counter() - started) * 1000.0,
                        tokens=accepted + int(suffix_ids.shape[-1]),
                        draft=stats,
                    )
            except Exception:
                return self._dflash_recompute(
                    inputs,
                    draft,
                    draft_ids,
                    chunk_size=chunk_size,
                    max_new_tokens=max_new_tokens,
                    started=started,
                )

        stats = DraftVerificationStats(
            mode="dflash_cache_verify",
            accepted=True,
            prefix_accepted=True,
            draft_tokens=len(draft_ids),
            checked_tokens=checked,
            matched_tokens=accepted,
            accepted_tokens=accepted,
            rejected_tokens=0,
            rollback_tokens=0,
            generated_tokens=0,
            chunk_size=max(1, chunk_size),
        )
        return VlmResult(
            text=draft,
            backend="dflash:accepted",
            ms=(time.perf_counter() - started) * 1000.0,
            tokens=accepted,
            draft=stats,
        )

    def _dflash_recompute(
        self,
        inputs: dict[str, object],
        draft: str,
        draft_ids: list[int],
        *,
        chunk_size: int,
        max_new_tokens: int,
        started: float,
    ) -> VlmResult:

        accepted = 0
        checked = 0
        with torch.no_grad():
            for expected in draft_ids:
                forward_inputs = append_token_ids(inputs, draft_ids[:accepted])
                outputs = self.model(**forward_inputs, use_cache=False)
                predicted = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
                checked += 1
                if predicted != expected:
                    generated_inputs = append_token_ids(inputs, draft_ids[:accepted])
                    outputs = self.model.generate(
                        **generation_inputs(generated_inputs),
                        max_new_tokens=max_new_tokens,
                    )
                    generated_ids = trim_prompt_ids(outputs, generated_inputs)[0]
                    text = decode_token_ids(
                        self.tokenizer,
                        draft_ids[:accepted] + tensor_to_ids(generated_ids),
                    )
                    stats = DraftVerificationStats(
                        mode="dflash_recompute_verify",
                        accepted=False,
                        prefix_accepted=accepted > 0,
                        draft_tokens=len(draft_ids),
                        checked_tokens=checked,
                        matched_tokens=accepted,
                        accepted_tokens=accepted,
                        rejected_tokens=len(draft_ids) - accepted,
                        rollback_tokens=1,
                        generated_tokens=int(generated_ids.shape[-1]),
                        chunk_size=max(1, chunk_size),
                    )
                    return VlmResult(
                        text=text,
                        backend="dflash:prefix"
                        if accepted
                        else "dflash:fallback-generate",
                        ms=(time.perf_counter() - started) * 1000.0,
                        tokens=accepted + int(generated_ids.shape[-1]),
                        draft=stats,
                    )
                accepted += 1

        stats = DraftVerificationStats(
            mode="dflash_recompute_verify",
            accepted=True,
            prefix_accepted=True,
            draft_tokens=len(draft_ids),
            checked_tokens=checked,
            matched_tokens=accepted,
            accepted_tokens=accepted,
            rejected_tokens=0,
            rollback_tokens=0,
            generated_tokens=0,
            chunk_size=max(1, chunk_size),
        )
        return VlmResult(
            text=draft,
            backend="dflash:accepted",
            ms=(time.perf_counter() - started) * 1000.0,
            tokens=accepted,
            draft=stats,
        )

    def _generate_suffix(
        self,
        inputs: dict[str, object],
        prefix_ids: list[int],
        max_new_tokens: int,
    ) -> tuple[object, str]:
        generated_inputs = append_token_ids(inputs, prefix_ids)
        outputs = self.model.generate(**generated_inputs, max_new_tokens=max_new_tokens)
        suffix_ids = trim_prompt_ids(outputs, generated_inputs)[0]
        suffix = decode_token_ids(self.tokenizer, tensor_to_ids(suffix_ids))
        return suffix_ids, suffix

    def _prefill(self, inputs: dict[str, object]):
        outputs = self.model(**inputs, use_cache=True, logits_to_keep=1)
        return outputs.logits, outputs.past_key_values

    def _forward_ids(self, token_ids: list[int], cache: object, like: object):
        return self.model(
            input_ids=tensor_from_ids(token_ids, like),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=len(token_ids),
        )

    def _decode_one(self, token_id: int, cache: object, like: object):
        outputs = self._forward_ids([token_id], cache, like)
        return outputs.logits, outputs.past_key_values

    def _continue_greedy(
        self,
        logits: object,
        cache: object,
        max_new_tokens: int,
        like: object,
    ) -> list[int]:

        generated: list[int] = []
        current_logits = logits
        current_cache = cache
        for _ in range(max(0, max_new_tokens)):
            token = int(torch.argmax(current_logits[:, -1, :], dim=-1).item())
            if is_eos(self.model, token):
                break
            generated.append(token)
            current_logits, current_cache = self._decode_one(token, current_cache, like)
        return generated


def patch_transformers_for_paddleocr_vl() -> None:
    try:
        from transformers import masking_utils
    except Exception:
        return
    original = getattr(masking_utils, "create_causal_mask", None)
    if (
        original is None
        or getattr(original, "__name__", "") == "_patched_create_causal_mask"
    ):
        return

    def _patched_create_causal_mask(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original(*args, **kwargs)

    masking_utils.create_causal_mask = _patched_create_causal_mask


def normalize_dflash_draft(text: str) -> str:
    out: list[str] = []
    chars = list(text)
    for index, ch in enumerate(chars):
        if ch.isspace():
            prev = next(
                (item for item in reversed(chars[:index]) if not item.isspace()), None
            )
            next_ = next(
                (item for item in chars[index + 1 :] if not item.isspace()), None
            )
            if should_drop_space(prev, next_):
                continue
            if not out or out[-1] not in {" ", "\n"}:
                out.append(" ")
            continue
        out.append(ch)
    return canonicalize_vlm_draft("".join(out))


def canonicalize_vlm_draft(text: str) -> str:
    text = THE_FIRST_LIGATURE_RE.sub("the first", text)
    text = BEEN_FIRMLY_LIGATURE_RE.sub("been firmly", text)
    text = (
        text.replace("∗", r"\(^{*}\)")
        .replace("†", r"\(^{\dagger}\)")
        .replace("‡", r"\(^{\ddagger}\)")
        .replace("Ł", "L")
        .replace("ł", "l")
    )
    text = unicodedata.normalize("NFKC", text)
    text = CITATION_RE.sub(lambda m: f" [{m.group(1).replace(' ', '')}] ", text)
    text = PUNCT_NO_SPACE_RE.sub(r"\1 ", text)
    text = SENTENCE_NO_SPACE_RE.sub(". ", text)
    text = INITIAL_SPLIT_RE.sub(". ", text)
    text = PAREN_NO_SPACE_RE.sub(" (", text)
    text = SUPERSCRIPT_NO_SPACE_RE.sub(r"\1 ", text)
    return MULTISPACE_RE.sub(" ", text)


def should_drop_space(prev: str | None, next_: str | None) -> bool:
    if prev is None or next_ is None:
        return True
    return (
        (is_cjk(prev) and next_.isascii() and next_.isdigit())
        or (prev.isascii() and prev.isdigit() and is_cjk(next_))
        or (is_cjk(prev) and next_ in {"%", ".", ",", "/", "-"})
        or (prev in {"%", ".", ",", "/", "-"} and is_cjk(next_))
    )


def tokenize_draft(tokenizer: object, draft: str) -> list[int]:
    encoded = tokenizer.encode(draft, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return list(encoded)


def split_chunks(
    tokenizer: object, draft_ids: list[int], chunk_size: int
) -> list[list[int]]:
    chunk_size = max(1, chunk_size)
    chunks: list[list[int]] = []
    start = 0
    while start < len(draft_ids):
        if is_special_chunk_token(tokenizer, draft_ids[start]):
            chunks.append([draft_ids[start]])
            start += 1
            continue
        end = start
        while (
            end < len(draft_ids)
            and end - start < chunk_size
            and not is_special_chunk_token(tokenizer, draft_ids[end])
        ):
            end += 1
        chunks.append(draft_ids[start:end])
        start = end
    return chunks


def is_special_chunk_token(tokenizer: object, token_id: int) -> bool:
    piece = decode_piece(tokenizer, token_id)
    return bool(piece) and all(
        ch.isspace() or ch in ",.;:!?%)]}，。；：！？、" for ch in piece
    )


def decode_piece(tokenizer: object, token_id: int) -> str:
    try:
        return str(tokenizer.decode([token_id], skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode([token_id]))


def chunk_predictions(prefix_logits: object, chunk_logits: object) -> list[int]:

    first = torch.argmax(prefix_logits[:, -1, :], dim=-1).reshape(1)
    if chunk_logits.shape[-2] <= 1:
        return [int(first.item())]
    rest = torch.argmax(chunk_logits[:, :-1, :], dim=-1).reshape(-1)
    return [int(item) for item in torch.cat([first, rest], dim=0).tolist()]


def cache_seq_length(cache: object) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if getter is not None:
        return int(getter())
    try:
        return int(cache[0][0].shape[-2])
    except Exception:
        return 0


def crop_cache(cache: object, max_length: int) -> None:
    crop = getattr(cache, "crop", None)
    if crop is None:
        raise RuntimeError("cache crop unsupported")
    crop(max_length)


def tensor_from_ids(token_ids: list[int], like: object):

    return torch.tensor([token_ids], dtype=like.dtype, device=like.device)


def append_token_ids(
    inputs: dict[str, object], token_ids: list[int]
) -> dict[str, object]:

    if not token_ids:
        return dict(inputs)
    suffix = tensor_from_ids(token_ids, inputs["input_ids"])
    out = dict(inputs)
    out["input_ids"] = torch.cat([inputs["input_ids"], suffix], dim=-1)
    if "attention_mask" in out:
        out["attention_mask"] = torch.cat(
            [
                out["attention_mask"],
                torch.ones_like(suffix, dtype=out["attention_mask"].dtype),
            ],
            dim=-1,
        )
    if "mm_token_type_ids" in out:
        out["mm_token_type_ids"] = torch.cat(
            [
                out["mm_token_type_ids"],
                torch.zeros_like(suffix, dtype=out["mm_token_type_ids"].dtype),
            ],
            dim=-1,
        )
    return out


def trim_prompt_ids(output_ids: object, inputs: dict[str, object]) -> object:
    return output_ids[:, int(inputs["input_ids"].shape[-1]) :]


def generation_inputs(inputs: dict[str, object]) -> dict[str, object]:
    return dict(inputs)


def decode_token_ids(tokenizer: object, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(token_ids))


def tensor_to_ids(value: torch.Tensor | list[torch.Tensor]) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [int(item) for item in value[0]]
        return [int(item) for item in value]
    return [int(value)]


def is_eos(model: object, token_id: int) -> bool:
    eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(eos, list):
        return token_id in eos
    return eos is not None and token_id == int(eos)


def is_cjk(ch: str) -> bool:
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


def main() -> None:
    import argparse
    import json

    from PIL import Image

    parser = argparse.ArgumentParser(
        description="Debug PaddleOCR-VL baseline generate and dflash_generate."
    )
    parser.add_argument("--image", type=Path, default=DEBUG_IMAGE)
    parser.add_argument("--model", type=Path, default=DEBUG_MODEL)
    parser.add_argument("--task", default=DEBUG_TASK, choices=sorted(PROMPTS))
    parser.add_argument("--mode", choices=["baseline", "dflash"], default=DEBUG_MODE)
    parser.add_argument("--draft", default=DEBUG_DRAFT)
    parser.add_argument("--draft-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    draft = args.draft
    if args.draft_file is not None:
        draft = args.draft_file.read_text(encoding="utf-8")

    with Image.open(args.image) as raw_image:
        image = raw_image.convert("RGB")
        runner = PaddleOCRVLRunner(
            args.model,
            device=args.device,
            dtype=args.dtype,
            max_pixels=args.max_pixels,
        )
        if args.mode == "baseline":
            result = runner.generate(image, args.task, max_new_tokens=args.max_tokens)
        else:
            result = runner.dflash_generate(
                image,
                args.task,
                draft,
                chunk_size=args.chunk_size,
                max_new_tokens=args.max_tokens,
            )

    print(result.text)
    print(
        json.dumps(
            {
                "backend": result.backend,
                "ms": result.ms,
                "tokens": result.tokens,
                "draft": result.draft,
            },
            default=lambda value: (
                value.__dict__ if hasattr(value, "__dict__") else str(value)
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
