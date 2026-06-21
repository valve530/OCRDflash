from __future__ import annotations

import time
from numbers import Real
from pathlib import Path
from typing import Any

from .io_utils import read_json, sha256_file, write_json
from .schemas import BoundingBox, DetectionResult, LayoutBlock


TEXT_NATIVE_CLASSES = {
    "text",
    "content",
    "paragraph_title",
    "doc_title",
    "abstract",
    "aside_text",
    "header",
    "footer",
    "footnote",
    "reference",
    "reference_content",
    "vision_footnote",
    "number",
    "vertical_text",
}


class LayoutDetector:
    def detect(self, image_path: Path, image_size: tuple[int, int]) -> DetectionResult:
        raise NotImplementedError


class JsonLayoutDetector(LayoutDetector):
    def __init__(self, layout_json: str | Path):
        self.layout_json = Path(layout_json)

    def detect(self, image_path: Path, image_size: tuple[int, int]) -> DetectionResult:
        _ = image_path, image_size
        return DetectionResult.from_obj(read_json(self.layout_json))


class WholePageLayoutDetector(LayoutDetector):
    def __init__(self, class_name: str = "text"):
        self.class_name = class_name

    def detect(self, image_path: Path, image_size: tuple[int, int]) -> DetectionResult:
        width, height = image_size
        return DetectionResult(
            schema_version=1,
            model_id="whole-page-layout",
            model_revision=None,
            threshold=0.0,
            image_size=[width, height],
            class_names=[self.class_name],
            blocks=[
                LayoutBlock(
                    bbox=BoundingBox(0.0, 0.0, float(width), float(height)),
                    score=1.0,
                    label=0,
                    class_name=self.class_name,
                )
            ],
            input_sha256=sha256_file(image_path) if image_path.exists() else None,
        )


class ProvidedBlocksLayoutDetector(LayoutDetector):
    def __init__(self, blocks: list[LayoutBlock], model_id: str = "provided-layout"):
        self.blocks = blocks
        self.model_id = model_id

    def detect(self, image_path: Path, image_size: tuple[int, int]) -> DetectionResult:
        width, height = image_size
        blocks = [
            LayoutBlock(
                bbox=block.bbox.clamp(width, height),
                score=block.score,
                label=block.label,
                class_name=block.class_name,
            )
            for block in self.blocks
        ]
        return DetectionResult(
            schema_version=1,
            model_id=self.model_id,
            model_revision=None,
            threshold=0.0,
            image_size=[width, height],
            class_names=sorted({block.class_name for block in blocks}) or ["text"],
            blocks=blocks,
            input_sha256=sha256_file(image_path) if image_path.exists() else None,
        )


class PaddleOCRLayoutDetector(LayoutDetector):
    def __init__(
        self,
        model_name: str = "PP-DocLayoutV2",
        device: str | None = None,
        threshold: float = 0.5,
        img_size: int | None = None,
        layout_nms: bool | None = None,
        layout_unclip_ratio: float | None = None,
        layout_merge_bboxes_mode: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.img_size = img_size
        self.layout_nms = layout_nms
        self.layout_unclip_ratio = layout_unclip_ratio
        self.layout_merge_bboxes_mode = layout_merge_bboxes_mode
        self._model = None

    def detect(self, image_path: Path, image_size: tuple[int, int]) -> DetectionResult:
        width, height = image_size
        model = self._load_model()
        raw = _run_paddleocr_layout_model(model, image_path)
        blocks = [
            block
            for block in (_layout_block_from_paddleocr(item, index, width, height) for index, item in enumerate(raw))
            if block is not None and block.score >= self.threshold
        ]
        blocks.sort(key=lambda block: (block.bbox.y0, block.bbox.x0))
        return DetectionResult(
            schema_version=1,
            model_id=f"paddleocr:{self.model_name}",
            model_revision=None,
            threshold=self.threshold,
            image_size=[width, height],
            class_names=sorted({block.class_name for block in blocks}) or ["text"],
            blocks=blocks,
            input_sha256=sha256_file(image_path) if image_path.exists() else None,
        )

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from paddleocr import LayoutDetection
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PaddleOCR layout mode requires the optional `paddleocr` package. "
                "Install it in the uv environment, or use `--layout-mode pdf-lines` / `--layout-json`."
            ) from exc

        kwargs: dict[str, object] = {}
        if self.img_size is not None:
            kwargs["img_size"] = self.img_size
        if self.threshold is not None:
            kwargs["threshold"] = self.threshold
        if self.layout_nms is not None:
            kwargs["layout_nms"] = self.layout_nms
        if self.layout_unclip_ratio is not None:
            kwargs["layout_unclip_ratio"] = self.layout_unclip_ratio
        if self.layout_merge_bboxes_mode is not None:
            kwargs["layout_merge_bboxes_mode"] = self.layout_merge_bboxes_mode
        if self.device:
            kwargs["device"] = self.device
            if self.device.startswith("cpu"):
                kwargs["enable_mkldnn"] = False
                kwargs["enable_cinn"] = False
        self._model = LayoutDetection(model_name=self.model_name, **kwargs)
        return self._model


def detect_layout(
    detector: LayoutDetector,
    image_path: str | Path,
    image_size: tuple[int, int],
    out_path: str | Path | None = None,
) -> tuple[DetectionResult, float]:
    started = time.perf_counter()
    result = detector.detect(Path(image_path), image_size)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if out_path is not None:
        write_json(out_path, result)
    return result, elapsed_ms


def should_use_native_text(class_name: str) -> bool:
    return class_name in TEXT_NATIVE_CLASSES


def _run_paddleocr_layout_model(model: object, image_path: Path) -> list[Any]:
    if hasattr(model, "predict"):
        result = model.predict(str(image_path))  # type: ignore[attr-defined]
    elif callable(model):
        result = model(str(image_path))  # type: ignore[operator]
    else:
        raise RuntimeError("PaddleOCR layout model is not callable and has no predict() method")

    if isinstance(result, dict):
        return _extract_layout_items(result)
    if isinstance(result, list):
        items: list[Any] = []
        for value in result:
            if isinstance(value, dict):
                extracted = _extract_layout_items(value)
                items.extend(extracted if extracted else [value])
            elif isinstance(value, list):
                items.extend(value)
            elif hasattr(value, "json"):
                try:
                    extracted = value.json
                    if isinstance(extracted, dict):
                        items.extend(_extract_layout_items(extracted) or [extracted])
                    else:
                        items.append(extracted)
                except Exception:
                    items.append(value)
            else:
                items.append(value)
        return items
    return list(result) if result is not None else []


def _extract_layout_items(value: dict[str, Any]) -> list[Any]:
    for key in ("boxes", "layout", "layout_result", "res", "dt_polys"):
        item = value.get(key)
        if isinstance(item, list):
            return item
    return []


def _layout_block_from_paddleocr(
    item: Any,
    index: int,
    image_width: int,
    image_height: int,
) -> LayoutBlock | None:
    if isinstance(item, dict):
        bbox_value = (
            item.get("bbox")
            or item.get("box")
            or item.get("coordinate")
            or item.get("poly")
            or item.get("points")
        )
        class_name = str(item.get("label") or item.get("class_name") or item.get("category") or "text")
        score = float(item.get("score") or item.get("confidence") or 1.0)
        label = int(item.get("class_id") or item.get("cls_id") or item.get("label_id") or index)
    elif isinstance(item, (list, tuple)) and len(item) >= 4:
        bbox_value = item[:4]
        class_name = "text"
        score = float(item[4]) if len(item) >= 5 and isinstance(item[4], (int, float)) else 1.0
        label = index
    else:
        return None

    bbox = _bbox_from_layout_value(bbox_value)
    if bbox is None:
        return None
    bbox = bbox.clamp(image_width, image_height)
    if bbox.width <= 1.0 or bbox.height <= 1.0:
        return None
    return LayoutBlock(bbox=bbox, score=score, label=label, class_name=_normalize_layout_class(class_name))


def _bbox_from_layout_value(value: Any) -> BoundingBox | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return BoundingBox.from_obj(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, Real) for item in value):
            return BoundingBox.from_obj(value)
        points: list[tuple[float, float]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                points.append((float(item[0]), float(item[1])))
        if points:
            return BoundingBox(
                x0=min(point[0] for point in points),
                y0=min(point[1] for point in points),
                x1=max(point[0] for point in points),
                y1=max(point[1] for point in points),
            )
    return None


def _normalize_layout_class(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "plain_text": "text",
        "title": "paragraph_title",
        "doc_title": "doc_title",
        "table_caption": "text",
        "figure_caption": "text",
        "image": "figure",
    }
    return aliases.get(normalized, normalized or "text")
