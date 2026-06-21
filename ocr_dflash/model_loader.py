from __future__ import annotations

from pathlib import Path
import inspect
import sys

from .generator import PaddleOCRVLDFlashGenerator, TransformersVlmGenerator


def load_transformers_vlm(
    model_id_or_path: str | Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    backend: str = "auto",
    attn_implementation: str | None = None,
    max_pixels: int | None = None,
) -> TransformersVlmGenerator | PaddleOCRVLDFlashGenerator:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    model_ref = str(model_id_or_path)
    _patch_paddleocr_vl_rope_alias()
    processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    dtype_value = _dtype(dtype)
    use_dflash = _prefer_paddleocr_vl(model_ref, processor, backend)
    kwargs = {"trust_remote_code": trust_remote_code}
    if dtype_value is not None:
        kwargs["dtype"] = dtype_value
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if device == "auto":
        kwargs["device_map"] = "auto"

    model_classes = [AutoModelForImageTextToText, AutoModelForCausalLM]

    errors: list[str] = []
    for cls in model_classes:
        try:
            model = cls.from_pretrained(model_ref, **kwargs)
            if device != "auto" and hasattr(model, "to"):
                model = model.to(device)
            model.eval()
            _patch_paddleocr_vl_compatibility(model)
            if use_dflash:
                return PaddleOCRVLDFlashGenerator(model=model, processor=processor, max_pixels=max_pixels)
            return TransformersVlmGenerator(model=model, processor=processor)
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError("failed to load VLM model:\n" + "\n".join(errors))


def _patch_paddleocr_vl_compatibility(model: object) -> None:
    _patch_paddleocr_vl_prepare_inputs(model)
    module_name = getattr(model.__class__, "__module__", "")
    module = sys.modules.get(module_name)
    if module is None or not hasattr(module, "create_causal_mask"):
        return
    original = getattr(module, "create_causal_mask")
    if getattr(original, "__name__", "") == "_compat_create_causal_mask":
        return
    params = inspect.signature(original).parameters

    def _compat_create_causal_mask(*args, **kwargs):
        if "cache_position" in kwargs and "cache_position" not in params:
            kwargs.pop("cache_position")
        if "inputs_embeds" in kwargs and "inputs_embeds" not in params and "input_embeds" in params:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        if "input_embeds" in kwargs and "input_embeds" not in params and "inputs_embeds" in params:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        return original(*args, **kwargs)

    setattr(module, "create_causal_mask", _compat_create_causal_mask)
    try:
        import sys as _sys

        module_obj = _sys.modules.get(module_name)
        if module_obj is not None and hasattr(module_obj, "create_causal_mask"):
            setattr(module_obj, "create_causal_mask", _compat_create_causal_mask)
    except Exception:
        pass


def _patch_paddleocr_vl_prepare_inputs(model: object) -> None:
    original = getattr(model, "prepare_inputs_for_generation", None)
    if original is None or getattr(original, "__name__", "") == "_compat_prepare_inputs_for_generation":
        return

    def _compat_prepare_inputs_for_generation(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except TypeError as exc:
            if "NoneType" not in str(exc) or "subscriptable" not in str(exc):
                raise
            input_ids = args[0] if args else kwargs.get("input_ids")
            if input_ids is None:
                raise
            try:
                import torch

                kwargs["cache_position"] = torch.arange(
                    int(input_ids.shape[-1]),
                    device=input_ids.device,
                    dtype=torch.long,
                )
            except Exception:
                raise exc
            return original(*args, **kwargs)

    setattr(model, "prepare_inputs_for_generation", _compat_prepare_inputs_for_generation)


def _patch_paddleocr_vl_rope_alias() -> None:
    try:
        from transformers import modeling_rope_utils
    except Exception:
        return
    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return
    if "proportional" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return
    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = modeling_rope_utils.ROPE_INIT_FUNCTIONS["proportional"]


def _dtype(value: str):
    if value == "auto":
        return None
    dtype_aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    canonical = dtype_aliases.get(value)
    if canonical is None:
        raise ValueError(f"unknown dtype: {value}")
    try:
        import torch

        return getattr(torch, canonical)
    except ModuleNotFoundError:
        return canonical


def _prefer_paddleocr_vl(model_ref: str, processor: object, backend: str) -> bool:
    if backend == "paddleocr-vl":
        return True
    if backend not in {"auto", "transformers"}:
        return False
    if "paddleocr-vl" in model_ref.lower():
        return True
    return "PaddleOCRVL" in processor.__class__.__name__
