from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamp(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x0=min(max(self.x0, 0.0), float(width)),
            y0=min(max(self.y0, 0.0), float(height)),
            x1=min(max(self.x1, 0.0), float(width)),
            y1=min(max(self.y1, 0.0), float(height)),
        )

    @classmethod
    def from_obj(cls, value: Any) -> "BoundingBox":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                x0=float(value["x0"]),
                y0=float(value["y0"]),
                x1=float(value["x1"]),
                y1=float(value["y1"]),
            )
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return cls(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        raise TypeError(f"cannot parse bbox from {value!r}")


@dataclass(slots=True)
class LayoutBlock:
    bbox: BoundingBox
    score: float
    label: int
    class_name: str

    @classmethod
    def from_obj(cls, value: dict[str, Any]) -> "LayoutBlock":
        return cls(
            bbox=BoundingBox.from_obj(value["bbox"]),
            score=float(value.get("score", 1.0)),
            label=int(value.get("label", 0)),
            class_name=str(value.get("class_name", value.get("class", "text"))),
        )


@dataclass(slots=True)
class DetectionResult:
    schema_version: int
    model_id: str
    model_revision: str | None
    threshold: float
    image_size: list[int]
    class_names: list[str]
    blocks: list[LayoutBlock]
    raw_logits: list[list[float]] | None = None
    raw_boxes: list[BoundingBox] | None = None
    input_sha256: str | None = None

    @classmethod
    def from_obj(cls, value: dict[str, Any]) -> "DetectionResult":
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            model_id=str(value.get("model_id", "external-layout-json")),
            model_revision=value.get("model_revision"),
            threshold=float(value.get("threshold", 0.0)),
            image_size=[int(value["image_size"][0]), int(value["image_size"][1])],
            class_names=[str(name) for name in value.get("class_names", ["text"])],
            blocks=[LayoutBlock.from_obj(block) for block in value.get("blocks", [])],
            raw_logits=value.get("raw_logits"),
            raw_boxes=[BoundingBox.from_obj(box) for box in value.get("raw_boxes", [])]
            if value.get("raw_boxes") is not None
            else None,
            input_sha256=value.get("input_sha256"),
        )


@dataclass(slots=True)
class RenderedPage:
    image_path: Path
    width: int
    height: int
    scale: float
    source: str


@dataclass(slots=True)
class NativeTextQuality:
    char_count: int
    char_area_ratio: float
    native_bbox_area_ratio: float
    width_coverage: float
    height_coverage: float
    line_count: int
    direct_accept: bool
    direct_accept_reason: str


@dataclass(slots=True)
class NativeTextCandidate:
    text: str
    quality: NativeTextQuality


@dataclass(slots=True)
class NativeTextBlockReport:
    index: int
    class_name: str
    bbox: BoundingBox
    text: str
    quality: NativeTextQuality | None


@dataclass(slots=True)
class NativeTextPageReport:
    schema_version: int
    pdf: str
    page: int
    dpi: int
    scale: float
    image_size: list[int]
    blocks: list[NativeTextBlockReport]


@dataclass(slots=True)
class DraftVerificationStats:
    mode: str
    accepted: bool
    prefix_accepted: bool
    draft_tokens: int
    checked_tokens: int
    matched_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    rollback_tokens: int
    generated_tokens: int
    chunk_size: int


@dataclass(slots=True)
class BlockRecognition:
    backend: str
    text: str
    tokens: int
    ms: float
    draft: DraftVerificationStats | None = None


@dataclass(slots=True)
class PageDemoBlock:
    index: int
    class_name: str
    label: int
    score: float
    bbox: BoundingBox
    native_text_draft: str | None
    native_text_quality: NativeTextQuality | None
    native_text: str | None
    recognition: BlockRecognition | None
    recognition_error: str | None
    crop: str | None = None


@dataclass(slots=True)
class PageArtifacts:
    layout_json: str
    native_text_json: str | None
    markdown: str
    rendered_image: str
    crops_dir: str | None = None


@dataclass(slots=True)
class PageReport:
    schema_version: int
    source: str
    image: str
    image_size: list[int]
    block_count: int
    layout_ms: float
    native_text_ms: float
    recognition_ms: float
    total_ms: float
    artifacts: PageArtifacts
    blocks: list[PageDemoBlock]
    config: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
