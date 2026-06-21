from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .pdf_text import DEFAULT_MAX_SIDE, _image_bbox_to_pdf_bbox, _render_scale
from .schemas import BoundingBox, LayoutBlock


@dataclass(slots=True)
class TableWord:
    text: str
    bbox: BoundingBox

    @property
    def cx(self) -> float:
        return (self.bbox.x0 + self.bbox.x1) * 0.5

    @property
    def cy(self) -> float:
        return (self.bbox.y0 + self.bbox.y1) * 0.5


def table_draft_from_pdf(
    pdf_path: str | Path,
    page_index: int,
    dpi: int,
    image_height: int,
    block: LayoutBlock,
) -> str | None:
    try:
        import fitz
    except ModuleNotFoundError:
        return None

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    scale = _render_scale(page.rect.width, page.rect.height, dpi, DEFAULT_MAX_SIDE)
    pdf_bbox = _image_bbox_to_pdf_bbox(block.bbox, image_height, scale)
    words = []
    for item in page.get_text("words"):
        x0, y0, x1, y1, text = item[:5]
        bbox = BoundingBox(float(x0), float(y0), float(x1), float(y1))
        if text and _center_inside(bbox, pdf_bbox):
            words.append(TableWord(str(text), bbox))
    if not words:
        return None

    rows = _cluster_rows(words)
    if not rows:
        return None
    col_centers = _cluster_columns_from_rows(rows)
    if not col_centers:
        return None

    rendered_rows: list[str] = []
    for row in rows:
        cells = _row_cells(row, col_centers)
        rendered_rows.append("".join(_render_cell(cell) for cell in cells))
    return "<nl>".join(rendered_rows)


def _cluster_rows(words: list[TableWord]) -> list[list[TableWord]]:
    median_h = sorted(word.bbox.height for word in words)[len(words) // 2]
    threshold = max(2.0, median_h * 0.65)
    rows: list[list[TableWord]] = []
    for word in sorted(words, key=lambda item: (item.cy, item.cx)):
        if not rows or abs(_row_center(rows[-1]) - word.cy) > threshold:
            rows.append([word])
        else:
            rows[-1].append(word)
    for row in rows:
        row.sort(key=lambda item: item.cx)
    return rows


def _cluster_columns(words: list[TableWord]) -> list[float]:
    median_w = sorted(word.bbox.width for word in words)[len(words) // 2]
    threshold = max(6.0, median_w * 0.75)
    centers: list[float] = []
    for word in sorted(words, key=lambda item: item.cx):
        if not centers or abs(centers[-1] - word.cx) > threshold:
            centers.append(word.cx)
        else:
            centers[-1] = (centers[-1] + word.cx) * 0.5
    return centers


def _cluster_columns_from_rows(rows: list[list[TableWord]]) -> list[float]:
    if not rows:
        return []
    widest = max(rows, key=len)
    if len(widest) >= 3:
        return [word.cx for word in widest]
    return _cluster_columns([word for row in rows for word in row])


def _row_cells(row: list[TableWord], col_centers: list[float]) -> list[str]:
    cells = [""] * len(col_centers)
    for word in row:
        col = min(range(len(col_centers)), key=lambda index: abs(col_centers[index] - word.cx))
        cells[col] = (cells[col] + " " + word.text).strip()
    while cells and not cells[-1]:
        cells.pop()
    return cells


def _row_center(row: list[TableWord]) -> float:
    return sum(word.cy for word in row) / max(len(row), 1)


def _center_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    cx = (inner.x0 + inner.x1) * 0.5
    cy = (inner.y0 + inner.y1) * 0.5
    return outer.x0 <= cx <= outer.x1 and outer.y0 <= cy <= outer.y1


def _canonical_table_cell(text: str) -> str:
    text = text.strip()
    replacements = {
        "dmodel": r"\(d_{model}\)",
        "dff": r"\(d_{ff}\)",
        "dk": r"\(d_k\)",
        "dv": r"\(d_v\)",
        "Pdrop": r"\(P_{drop}\)",
        "ϵls": r"\(\epsilon_{ls}\)",
        "×106": r"\(\times 10^{6}\)",
    }
    return replacements.get(text, text)


def _render_cell(text: str) -> str:
    text = _canonical_table_cell(text)
    return "<fcel>" + text if text else "<ecel>"
