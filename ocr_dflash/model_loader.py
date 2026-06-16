from __future__ import annotations

from pathlib import Path

import torch

from .generator import PaddleOCRVLDFlashGenerator, TransformersVlmGenerator


def load_transformers_vlm(
    model_id_or_path: str | Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    backend: str = "auto",
) -> TransformersVlmGenerator | PaddleOCRVLDFlashGenerator:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    try:
        from transformers import PaddleOCRVLForConditionalGeneration
    except ImportError:  # pragma: no cover - older transformers or test doubles
        PaddleOCRVLForConditionalGeneration = None

    model_ref = str(model_id_or_path)
    processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    dtype_value = _dtype(dtype)
    kwargs = {"trust_remote_code": trust_remote_code}
    if dtype_value is not None:
        kwargs["dtype"] = dtype_value
    if device == "auto":
        kwargs["device_map"] = "auto"

    preferred = (
        PaddleOCRVLForConditionalGeneration
        if PaddleOCRVLForConditionalGeneration is not None and _prefer_paddleocr_vl(model_ref, processor, backend)
        else None
    )
    model_classes = [preferred] if preferred is not None else []
    model_classes.extend([AutoModelForImageTextToText, AutoModelForCausalLM])

    errors: list[str] = []
    for cls in model_classes:
        try:
            model = cls.from_pretrained(model_ref, **kwargs)
            if device != "auto" and hasattr(model, "to"):
                model = model.to(device)
            model.eval()
            if PaddleOCRVLForConditionalGeneration is not None and cls is PaddleOCRVLForConditionalGeneration:
                return PaddleOCRVLDFlashGenerator(model=model, processor=processor)
            return TransformersVlmGenerator(model=model, processor=processor)
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError("failed to load VLM model:\n" + "\n".join(errors))


def _dtype(value: str) -> torch.dtype | None:
    if value == "auto":
        return None
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype: {value}")


def _prefer_paddleocr_vl(model_ref: str, processor: object, backend: str) -> bool:
    if backend == "paddleocr-vl":
        return True
    if backend not in {"auto", "transformers"}:
        return False
    if "paddleocr-vl" in model_ref.lower():
        return True
    return "PaddleOCRVL" in processor.__class__.__name__
