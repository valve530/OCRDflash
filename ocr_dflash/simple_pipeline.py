from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .io_utils import ensure_dir, write_json
from .layout import PaddleOCRLayoutDetector, detect_layout, should_use_native_text
from .markdown_writer import write_markdown
from .pdf_text import extract_native_text_candidates_for_blocks, render_pdf_page_to_png
from .schemas import (
    BlockRecognition,
    DetectionResult,
    LayoutBlock,
    NativeTextCandidate,
    PageArtifacts,
    PageDemoBlock,
    PageReport,
    PdfReport,
    RenderedPage,
)
from .simple_vlm import PaddleOCRVLRunner


@dataclass(slots=True)
class SimpleOptions:
    pdf: Path
    out_dir: Path = Path("tmp/simple_out")
    dpi: int = 200
    layout_model_dir: Path = Path("models/PP-DocLayoutV3")
    vlm_model: Path = Path("models/PaddleOCR-VL-1.6")
    mode: str = "dflash"
    max_tokens: int = 512
    chunk_size: int = 8
    max_pixels: int = 1280 * 28 * 28


@dataclass(slots=True)
class PreparedPage:
    page: int
    rendered: RenderedPage
    layout: DetectionResult
    native_candidates: list[NativeTextCandidate | None]
    layout_ms: float
    native_ms: float
    layout_path: Path
    native_path: Path


def run_pdf(options: SimpleOptions) -> PdfReport:
    import fitz

    out_dir = ensure_dir(options.out_dir)
    started = time.perf_counter()
    doc = fitz.open(options.pdf)
    layout_detector = PaddleOCRLayoutDetector(
        model_name="PP-DocLayoutV3",
        model_dir=options.layout_model_dir,
        device="gpu",
        threshold=0.5,
        layout_nms=True,
    )

    prepared = [
        prepare_page(options, page_index, out_dir / f"page_{page_index:04}", layout_detector)
        for page_index in range(len(doc))
    ]

    reset_cuda_peak_memory_stats()
    runner = PaddleOCRVLRunner(options.vlm_model, device="cuda", dtype="bf16", max_pixels=options.max_pixels)
    reports = [recognize_page(options, page, runner, started) for page in prepared]
    peak_vram_mb, current_vram_mb = cuda_memory_stats()
    report = PdfReport(
        schema_version=1,
        source=str(options.pdf),
        page_count=len(reports),
        total_layout_ms=sum(page.layout_ms for page in prepared),
        total_native_text_ms=sum(page.native_ms for page in prepared),
        total_native_direct_ms=0.0,
        total_vlm_ms=sum(page.vlm_ms for page in reports),
        total_ms=(time.perf_counter() - started) * 1000.0,
        peak_vram_mb=peak_vram_mb,
        avg_vram_mb=current_vram_mb,
        pages=reports,
        config={
            "mode": options.mode,
            "dpi": options.dpi,
            "layout_model": str(options.layout_model_dir),
            "vlm_model": str(options.vlm_model),
            "max_tokens": options.max_tokens,
            "chunk_size": options.chunk_size,
            "max_pixels": options.max_pixels,
        },
        stats=pdf_stats(reports),
    )
    write_json(out_dir / "report.json", report)
    return report


def prepare_page(
    options: SimpleOptions,
    page_index: int,
    out_dir: Path,
    layout_detector: PaddleOCRLayoutDetector,
) -> PreparedPage:
    page_dir = ensure_dir(out_dir)
    rendered = render_pdf_page_to_png(options.pdf, page_index, options.dpi, page_dir / f"page_{page_index:04}.png")
    layout_path = page_dir / "layout.json"
    layout, layout_ms = detect_layout(
        layout_detector,
        rendered.image_path,
        (rendered.width, rendered.height),
        layout_path,
    )
    native_path = page_dir / "native_text.json"
    started_native = time.perf_counter()
    native_candidates = extract_native_text_candidates_for_blocks(
        options.pdf,
        page_index,
        options.dpi,
        rendered.width,
        rendered.height,
        layout.blocks,
        native_path,
    )
    native_ms = (time.perf_counter() - started_native) * 1000.0
    return PreparedPage(
        page=page_index,
        rendered=rendered,
        layout=layout,
        native_candidates=native_candidates,
        layout_ms=layout_ms,
        native_ms=native_ms,
        layout_path=layout_path,
        native_path=native_path,
    )


def recognize_page(
    options: SimpleOptions,
    prepared: PreparedPage,
    runner: PaddleOCRVLRunner,
    started_total: float,
) -> PageReport:
    from PIL import Image

    page_dir = prepared.rendered.image_path.parent
    crops_dir = ensure_dir(page_dir / "crops")
    blocks: list[PageDemoBlock] = []
    vlm_ms = 0.0
    with Image.open(prepared.rendered.image_path) as raw_image:
        image = raw_image.convert("RGB")
        for index, block in enumerate(prepared.layout.blocks):
            native = prepared.native_candidates[index] if index < len(prepared.native_candidates) else None
            crop_path = crops_dir / f"block_{index + 1:04}.png"
            crop = crop_image(image, block, crop_path)
            try:
                if options.mode == "baseline":
                    result = runner.generate(crop, block.class_name, max_new_tokens=options.max_tokens)
                else:
                    result = runner.dflash_generate(
                        crop,
                        block.class_name,
                        native.text if native else None,
                        chunk_size=options.chunk_size,
                        max_new_tokens=options.max_tokens,
                    )
                recognition = BlockRecognition(
                    backend=result.backend,
                    text=result.text,
                    tokens=result.tokens,
                    ms=result.ms,
                    draft=result.draft,
                )
                error = None
                vlm_ms += result.ms
            except Exception as exc:
                recognition = None
                error = str(exc)
            blocks.append(
                PageDemoBlock(
                    index=index + 1,
                    class_name=block.class_name,
                    label=block.label,
                    score=block.score,
                    bbox=block.bbox,
                    native_text_draft=native.text if native else None,
                    native_text_quality=native.quality if native else None,
                    native_text=None,
                    recognition=recognition,
                    recognition_error=error,
                    crop=str(crop_path),
                )
            )

    markdown_path = page_dir / "page.md"
    write_markdown(
        markdown_path,
        str(options.pdf),
        prepared.rendered.image_path,
        (prepared.rendered.width, prepared.rendered.height),
        blocks,
    )
    report = PageReport(
        schema_version=1,
        source=str(options.pdf),
        image=str(prepared.rendered.image_path),
        image_size=[prepared.rendered.width, prepared.rendered.height],
        block_count=len(blocks),
        layout_ms=prepared.layout_ms,
        native_text_ms=prepared.native_ms,
        native_direct_ms=0.0,
        vlm_ms=vlm_ms,
        recognition_ms=vlm_ms,
        total_ms=(time.perf_counter() - started_total) * 1000.0,
        artifacts=PageArtifacts(
            layout_json=str(prepared.layout_path),
            native_text_json=str(prepared.native_path),
            markdown=str(markdown_path),
            rendered_image=str(prepared.rendered.image_path),
            crops_dir=str(crops_dir),
        ),
        blocks=blocks,
        config={
            "page": prepared.page,
            "mode": options.mode,
            "dpi": options.dpi,
            "max_tokens": options.max_tokens,
            "chunk_size": options.chunk_size,
        },
        stats=page_stats(blocks),
    )
    write_json(page_dir / "report.json", report)
    return report


def crop_image(image: object, block: LayoutBlock, crop_path: Path) -> object:
    crop = image.crop((block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)).convert("RGB")
    crop.save(crop_path)
    return crop


def copy_pdf_outputs(report: PdfReport, out_dir: Path) -> None:
    if not report.pages:
        return
    first = report.pages[0]
    for src, name in (
        (first.artifacts.markdown, "page.md"),
        (first.artifacts.layout_json, "layout.json"),
        (first.artifacts.native_text_json, "native_text.json"),
        (first.artifacts.rendered_image, "page_0000.png"),
    ):
        if src:
            path = Path(src)
            if path.exists() and path.parent != out_dir:
                shutil.copyfile(path, out_dir / name)


def page_stats(blocks: list[PageDemoBlock]) -> dict[str, float | int]:
    draft_blocks = 0
    draft_tokens = 0
    accepted_tokens = 0
    rejected_tokens = 0
    generated_tokens = 0
    accepted_blocks = 0
    prefix_blocks = 0
    fallback_blocks = 0
    for block in blocks:
        if block.native_text_draft:
            draft_blocks += 1
        if block.recognition is None:
            continue
        if "fallback" in block.recognition.backend:
            fallback_blocks += 1
        stats = block.recognition.draft
        if stats is None:
            continue
        draft_tokens += stats.draft_tokens
        accepted_tokens += stats.accepted_tokens
        rejected_tokens += stats.rejected_tokens
        generated_tokens += stats.generated_tokens
        if stats.accepted:
            accepted_blocks += 1
        elif stats.prefix_accepted:
            prefix_blocks += 1
    return {
        "total_blocks": len(blocks),
        "native_eligible_blocks": sum(1 for block in blocks if should_use_native_text(block.class_name)),
        "draft_candidate_blocks": draft_blocks,
        "recognized_blocks": sum(1 for block in blocks if block.recognition is not None),
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "rejected_tokens": rejected_tokens,
        "generated_tokens": generated_tokens,
        "accepted_token_ratio": accepted_tokens / draft_tokens if draft_tokens else 0.0,
        "direct_accept_blocks": accepted_blocks,
        "prefix_accept_blocks": prefix_blocks,
        "fallback_blocks": fallback_blocks,
    }


def pdf_stats(reports: list[PageReport]) -> dict[str, float | int]:
    draft_tokens = sum(report.stats.get("draft_tokens", 0) for report in reports)
    accepted_tokens = sum(report.stats.get("accepted_tokens", 0) for report in reports)
    return {
        "page_count": len(reports),
        "block_count": sum(report.block_count for report in reports),
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "accepted_token_ratio": accepted_tokens / draft_tokens if draft_tokens else 0.0,
        "direct_accept_blocks": sum(report.stats.get("direct_accept_blocks", 0) for report in reports),
        "prefix_accept_blocks": sum(report.stats.get("prefix_accept_blocks", 0) for report in reports),
        "fallback_blocks": sum(report.stats.get("fallback_blocks", 0) for report in reports),
    }


def cuda_memory_stats() -> tuple[float | None, float | None]:
    try:
        import torch
    except Exception:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    return (
        float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        float(torch.cuda.memory_allocated() / (1024 * 1024)),
    )


def reset_cuda_peak_memory_stats() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
