from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    fitz = None

from .schemas import (
    BoundingBox,
    LayoutBlock,
    NativeTextBlockReport,
    NativeTextCandidate,
    NativeTextPageReport,
    NativeTextQuality,
    RenderedPage,
    to_jsonable,
)

DEFAULT_MAX_SIDE = 3500


@dataclass(slots=True)
class _CharBox:
    index: int
    ch: str
    bbox: BoundingBox


def render_pdf_page_to_png(
    pdf_path: str | Path,
    page_index: int,
    dpi: int,
    out_path: str | Path,
) -> RenderedPage:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDF rendering; install the project with uv sync")

    pdf_path = Path(pdf_path)
    out_path = Path(out_path)
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    page_rect = page.rect
    scale = _render_scale(page_rect.width, page_rect.height, dpi, DEFAULT_MAX_SIDE)
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(out_path)
    return RenderedPage(
        image_path=out_path,
        width=pix.width,
        height=pix.height,
        scale=scale,
        source=str(pdf_path),
    )


def extract_native_text_candidates_for_blocks(
    pdf_path: str | Path,
    page_index: int,
    dpi: int,
    image_width: int,
    image_height: int,
    blocks: list[LayoutBlock],
    out_path: str | Path | None = None,
) -> list[NativeTextCandidate | None]:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for PDF native text extraction; install the project with uv sync"
        )

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    page_rect = page.rect
    scale = _render_scale(page_rect.width, page_rect.height, dpi, DEFAULT_MAX_SIDE)
    chars = _page_chars(page)

    results: list[NativeTextCandidate | None] = []
    report_blocks: list[NativeTextBlockReport] = []
    for index, block in enumerate(blocks):
        pdf_bbox = _image_bbox_to_pdf_bbox(block.bbox, image_height, scale)
        selected = _select_chars_for_bbox(chars, pdf_bbox)
        text = _normalize_native_text(_collect_text_from_chars(selected))
        quality = _native_text_quality(block, pdf_bbox, selected) if text else None
        candidate = NativeTextCandidate(text=text, quality=quality) if quality else None
        results.append(candidate)
        report_blocks.append(
            NativeTextBlockReport(
                index=index + 1,
                class_name=block.class_name,
                bbox=block.bbox,
                text=text,
                quality=quality,
            )
        )

    if out_path is not None:
        report = NativeTextPageReport(
            schema_version=1,
            pdf=str(pdf_path),
            page=page_index,
            dpi=dpi,
            scale=scale,
            image_size=[image_width, image_height],
            blocks=report_blocks,
        )
        Path(out_path).write_text(
            __import__("json").dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return results


def detect_pdf_text_line_blocks(
    pdf_path: str | Path,
    page_index: int,
    dpi: int,
    image_width: int,
    image_height: int,
) -> list[LayoutBlock]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDF text line layout; install the project with uv sync")

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    page_rect = page.rect
    scale = _render_scale(page_rect.width, page_rect.height, dpi, DEFAULT_MAX_SIDE)

    blocks: list[LayoutBlock] = []
    for item in page.get_text("dict").get("blocks", []):
        if item.get("type", 0) != 0:
            continue
        for line in item.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue
            bbox = BoundingBox.from_obj(line.get("bbox", item.get("bbox", [0, 0, 0, 0])))
            image_bbox = _pdf_bbox_to_image_bbox(bbox, scale).clamp(image_width, image_height)
            if image_bbox.width < 1.0 or image_bbox.height < 1.0:
                continue
            blocks.append(
                LayoutBlock(
                    bbox=_pad_bbox(image_bbox, image_width, image_height),
                    score=1.0,
                    label=0,
                    class_name="text",
                )
            )

    blocks.sort(key=lambda block: (block.bbox.y0, block.bbox.x0))
    return _merge_nearby_line_blocks(blocks, image_width, image_height)


def extract_native_text_for_blocks(
    pdf_path: str | Path,
    page_index: int,
    dpi: int,
    image_width: int,
    image_height: int,
    blocks: list[LayoutBlock],
    out_path: str | Path | None = None,
) -> list[str | None]:
    candidates = extract_native_text_candidates_for_blocks(
        pdf_path,
        page_index,
        dpi,
        image_width,
        image_height,
        blocks,
        out_path,
    )
    return [candidate.text if candidate else None for candidate in candidates]


def _render_scale(page_width: float, page_height: float, dpi: int, max_side: int) -> float:
    scale = dpi / 72.0
    long_side = max(page_width, page_height)
    if long_side * scale > max_side:
        scale = max_side / long_side
    return scale


def _page_chars(page: fitz.Page) -> list[_CharBox]:
    chars: list[_CharBox] = []
    blocks = page.get_text("rawdict").get("blocks", [])
    index = 0
    for block in blocks:
        for line in block.get("lines", []):
            line_boxes: list[BoundingBox] = []
            for span in line.get("spans", []):
                span_chars = span.get("chars")
                if span_chars:
                    for item in span_chars:
                        ch = item.get("c", "")
                        if not ch or ch == "\0":
                            continue
                        chars.append(
                            _CharBox(
                                index=index,
                                ch=ch,
                                bbox=BoundingBox.from_obj(item.get("bbox", [0, 0, 0, 0])),
                            )
                        )
                        line_boxes.append(chars[-1].bbox)
                        index += 1
                    continue

                text = span.get("text", "")
                bbox = BoundingBox.from_obj(span.get("bbox", [0, 0, 0, 0]))
                if not text:
                    continue
                if len(text) == 1:
                    chars.append(_CharBox(index=index, ch=text, bbox=bbox))
                    line_boxes.append(bbox)
                    index += 1
                    continue
                step = bbox.width / max(len(text), 1)
                for offset, ch in enumerate(text):
                    char_box = BoundingBox(
                        x0=bbox.x0 + step * offset,
                        y0=bbox.y0,
                        x1=min(bbox.x0 + step * (offset + 1), bbox.x1),
                        y1=bbox.y1,
                    )
                    chars.append(_CharBox(index=index, ch=ch, bbox=char_box))
                    line_boxes.append(char_box)
                    index += 1
            if line_boxes:
                chars.append(_CharBox(index=index, ch="\n", bbox=_union_bbox(line_boxes) or line_boxes[-1]))
                index += 1
    return chars


def _image_bbox_to_pdf_bbox(bbox: BoundingBox, image_height: int, scale: float) -> BoundingBox:
    _ = image_height
    return BoundingBox(
        x0=bbox.x0 / scale,
        y0=bbox.y0 / scale,
        x1=bbox.x1 / scale,
        y1=bbox.y1 / scale,
    )


def _pdf_bbox_to_image_bbox(bbox: BoundingBox, scale: float) -> BoundingBox:
    return BoundingBox(
        x0=bbox.x0 * scale,
        y0=bbox.y0 * scale,
        x1=bbox.x1 * scale,
        y1=bbox.y1 * scale,
    )


def _pad_bbox(bbox: BoundingBox, image_width: int, image_height: int) -> BoundingBox:
    pad_x = max(2.0, bbox.height * 0.20)
    pad_y = max(2.0, bbox.height * 0.25)
    return BoundingBox(
        x0=bbox.x0 - pad_x,
        y0=bbox.y0 - pad_y,
        x1=bbox.x1 + pad_x,
        y1=bbox.y1 + pad_y,
    ).clamp(image_width, image_height)


def _merge_nearby_line_blocks(
    blocks: list[LayoutBlock],
    image_width: int,
    image_height: int,
) -> list[LayoutBlock]:
    if not blocks:
        return [
            LayoutBlock(
                bbox=BoundingBox(0.0, 0.0, float(image_width), float(image_height)),
                score=1.0,
                label=0,
                class_name="text",
            )
        ]

    merged: list[LayoutBlock] = []
    current = blocks[0]
    for block in blocks[1:]:
        current_height = max(current.bbox.height, 1.0)
        vertical_gap = block.bbox.y0 - current.bbox.y1
        horizontally_related = block.bbox.x0 <= current.bbox.x1 + current_height * 2.0
        same_paragraph = 0.0 <= vertical_gap <= current_height * 0.65 and horizontally_related
        if same_paragraph:
            current = LayoutBlock(
                bbox=BoundingBox(
                    x0=min(current.bbox.x0, block.bbox.x0),
                    y0=min(current.bbox.y0, block.bbox.y0),
                    x1=max(current.bbox.x1, block.bbox.x1),
                    y1=max(current.bbox.y1, block.bbox.y1),
                ).clamp(image_width, image_height),
                score=min(current.score, block.score),
                label=current.label,
                class_name=current.class_name,
            )
        else:
            merged.append(current)
            current = block
    merged.append(current)
    return merged


def _select_chars_for_bbox(chars: Iterable[_CharBox], bbox: BoundingBox) -> list[_CharBox]:
    selected = [
        ch
        for ch in chars
        if _overlap_ratio(ch.bbox, bbox) >= 0.2 or _center_inside(ch.bbox, bbox)
    ]
    selected.sort(key=lambda ch: ch.index)
    return selected


def _collect_text_from_chars(selected: list[_CharBox]) -> str:
    out: list[str] = []
    prev: _CharBox | None = None
    for ch in selected:
        if _should_insert_inferred_space(prev, ch):
            out.append(" ")
        out.append(ch.ch)
        prev = ch
    return "".join(out)


def _native_text_quality(
    block: LayoutBlock,
    block_pdf_bbox: BoundingBox,
    selected: list[_CharBox],
) -> NativeTextQuality:
    ink_chars = [ch for ch in selected if not ch.ch.isspace()]
    char_count = len(ink_chars)
    block_area = max(block_pdf_bbox.area, 1e-3)
    char_area = sum(min(ch.bbox.area, block_area) for ch in ink_chars)
    native_bbox = _union_bbox(ch.bbox for ch in ink_chars)
    if native_bbox is None:
        native_bbox_area_ratio = width_coverage = height_coverage = 0.0
        line_count = 0
    else:
        native_bbox_area_ratio = min(max(native_bbox.area / block_area, 0.0), 1.0)
        width_coverage = min(max(native_bbox.width / max(block_pdf_bbox.width, 1e-3), 0.0), 1.0)
        height_coverage = min(max(native_bbox.height / max(block_pdf_bbox.height, 1e-3), 0.0), 1.0)
        line_count = _estimate_line_count(selected)
    char_area_ratio = min(max(char_area / block_area, 0.0), 1.0)
    direct_accept, reason = _direct_accept_native_text(
        block.class_name,
        char_count,
        char_area_ratio,
        native_bbox_area_ratio,
        width_coverage,
        height_coverage,
        line_count,
    )
    return NativeTextQuality(
        char_count=char_count,
        char_area_ratio=char_area_ratio,
        native_bbox_area_ratio=native_bbox_area_ratio,
        width_coverage=width_coverage,
        height_coverage=height_coverage,
        line_count=line_count,
        direct_accept=direct_accept,
        direct_accept_reason=reason,
    )


def _direct_accept_native_text(
    class_name: str,
    char_count: int,
    char_area_ratio: float,
    native_bbox_area_ratio: float,
    width_coverage: float,
    height_coverage: float,
    line_count: int,
) -> tuple[bool, str]:
    if char_count == 0:
        return False, "empty-native-text"
    if not _is_direct_native_class(class_name):
        return False, "class-not-direct-native"

    short_text = char_count <= 12
    enough_ink = char_area_ratio >= (0.018 if short_text else 0.025)
    enough_bbox = native_bbox_area_ratio >= (0.08 if short_text else 0.12)
    enough_width = width_coverage >= (0.20 if short_text else 0.45)
    enough_height = height_coverage >= (0.28 if short_text else 0.35)
    enough_lines = line_count > 0

    if enough_ink and enough_bbox and enough_width and enough_height and enough_lines:
        return True, "quality-thresholds-passed"

    failed: list[str] = []
    if not enough_ink:
        failed.append("char-area")
    if not enough_bbox:
        failed.append("native-bbox-area")
    if not enough_width:
        failed.append("width-coverage")
    if not enough_height:
        failed.append("height-coverage")
    if not enough_lines:
        failed.append("line-count")
    return False, ",".join(failed)


def _is_direct_native_class(class_name: str) -> bool:
    return class_name in {
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
        "number",
        "vertical_text",
    }


def _normalize_native_text(text: str) -> str:
    out: list[str] = []
    pending_space = False
    for ch in text:
        if ch.isspace():
            pending_space = True
            continue
        if pending_space and _should_keep_native_space(out[-1][-1] if out else None, ch):
            out.append(" ")
        out.append(ch)
        pending_space = False
    return "".join(out).strip()


def _should_keep_native_space(prev: str | None, next_: str) -> bool:
    if prev is None:
        return False
    return (
        (prev.isascii() and prev.isalnum() and next_.isascii() and next_.isalnum())
        or (prev.isascii() and prev.isalnum() and _is_cjk(next_))
        or (_is_cjk(prev) and next_.isascii() and next_.isalnum())
    )


def _should_insert_inferred_space(prev: _CharBox | None, next_: _CharBox) -> bool:
    if prev is None:
        return False
    if prev.ch.isspace() or next_.ch.isspace():
        return False
    if not _should_keep_inferred_space(prev.ch, next_.ch):
        return False
    prev_height = max(prev.bbox.height, 1.0)
    next_height = max(next_.bbox.height, 1.0)
    same_line = abs(_center_y(prev.bbox) - _center_y(next_.bbox)) <= max(prev_height, next_height) * 0.5
    if not same_line:
        return False
    gap = next_.bbox.x0 - prev.bbox.x1
    if gap <= 0.0:
        return False
    avg_width = max((prev.bbox.width + next_.bbox.width) * 0.5, 1.0)
    return gap >= avg_width * 0.18


def _should_keep_inferred_space(prev: str, next_: str) -> bool:
    return (
        prev.isascii()
        and prev.isalpha()
        and _is_cjk(next_)
        or _is_cjk(prev)
        and next_.isascii()
        and next_.isalpha()
    )


def _estimate_line_count(selected: list[_CharBox]) -> int:
    line_centers: list[float] = []
    avg_height = (
        sum(max(ch.bbox.height, 1.0) for ch in selected) / len(selected) if selected else 1.0
    )
    threshold = avg_height * 0.65
    for ch in selected:
        y = _center_y(ch.bbox)
        if all(abs(existing - y) > threshold for existing in line_centers):
            line_centers.append(y)
    return len(line_centers)


def _overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    left = max(a.x0, b.x0)
    right = min(a.x1, b.x1)
    top = min(a.y1, b.y1)
    bottom = max(a.y0, b.y0)
    if right <= left or top <= bottom:
        return 0.0
    intersection = (right - left) * (top - bottom)
    area = max(a.area, 1e-3)
    return intersection / area


def _center_inside(a: BoundingBox, b: BoundingBox) -> bool:
    x = (a.x0 + a.x1) * 0.5
    y = (a.y0 + a.y1) * 0.5
    return b.x0 <= x <= b.x1 and b.y0 <= y <= b.y1


def _center_y(bbox: BoundingBox) -> float:
    return (bbox.y0 + bbox.y1) * 0.5


def _union_bbox(boxes: Iterable[BoundingBox]) -> BoundingBox | None:
    items = list(boxes)
    if not items:
        return None
    return BoundingBox(
        x0=min(box.x0 for box in items),
        y0=min(box.y0 for box in items),
        x1=max(box.x1 for box in items),
        y1=max(box.y1 for box in items),
    )


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )
