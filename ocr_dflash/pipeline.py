from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .generator import BlockGenerator, GenerationOptions, NativeDraftGenerator
from .io_utils import ensure_dir, write_json
from .layout import (
    JsonLayoutDetector,
    PaddleOCRLayoutDetector,
    ProvidedBlocksLayoutDetector,
    WholePageLayoutDetector,
    detect_layout,
    should_use_native_text,
)
from .model_loader import load_transformers_vlm
from .markdown_writer import write_markdown
from .pdf_text import detect_pdf_text_line_blocks, extract_native_text_candidates_for_blocks, render_pdf_page_to_png
from .schemas import (
    BlockRecognition,
    NativeTextCandidate,
    DetectionResult,
    PageArtifacts,
    PageDemoBlock,
    PageReport,
    PdfReport,
    RenderedPage,
    LayoutBlock,
)


@dataclass(slots=True)
class PipelineOptions:
    pdf: Path | None = None
    image: Path | None = None
    page: int = 0
    dpi: int = 200
    out_dir: Path = Path("out")
    layout_json: Path | None = None
    layout_mode: Literal["whole-page", "pdf-lines", "pp-doclayout-v2"] = "whole-page"
    layout_device: str | None = None
    layout_threshold: float = 0.5
    chunk_size: int = 16
    batch_size: int = 128
    batch_max_pixels: int = 0
    max_tokens: int = 256
    temperature: float = 0.0
    sampling: bool = False
    draft_mode: Literal["native", "none"] = "native"
    verify_native_text: bool = False
    fallback_to_native: bool = True
    generator: BlockGenerator | None = None
    vlm_model: Path | str | None = None
    vlm_device: str = "auto"
    vlm_dtype: str = "auto"
    vlm_backend: Literal["auto", "transformers", "paddleocr-vl"] = "auto"
    trust_remote_code: bool = True
    prompt: str = "Convert this document image region to Markdown."


@dataclass(slots=True)
class PreparedPage:
    page: int
    rendered: RenderedPage
    layout: DetectionResult
    layout_ms: float
    native_candidates: list[NativeTextCandidate | None]
    native_ms: float
    layout_path: Path
    native_path: Path


def run_page_pipeline(options: PipelineOptions) -> PageReport:
    prepared = prepare_page(options)
    generator = _build_generator(options)
    report = recognize_prepared_page(options, prepared, generator)
    write_json(ensure_dir(options.out_dir) / "report.json", report)
    return report


def run_pdf_pipeline(options: PipelineOptions) -> PdfReport:
    if options.pdf is None:
        raise ValueError("--pdf is required for run_pdf_pipeline")

    import fitz

    doc = fitz.open(options.pdf)
    page_count = len(doc)
    started_total = time.perf_counter()
    prepared_pages = [prepare_page(_page_options(options, page_index)) for page_index in range(page_count)]
    if _has_cuda():
        _reset_cuda_peak_memory_stats()
    generator = _build_generator(options)
    reports = recognize_prepared_pages(options, prepared_pages, generator, started_total=started_total)
    peak_vram_mb, avg_vram_mb = _cuda_memory_stats()
    report = PdfReport(
        schema_version=1,
        source=str(options.pdf),
        page_count=page_count,
        total_layout_ms=sum(page.layout_ms for page in prepared_pages),
        total_native_text_ms=sum(page.native_ms for page in prepared_pages),
        total_native_direct_ms=sum(report.native_direct_ms for report in reports),
        total_vlm_ms=sum(report.vlm_ms for report in reports),
        total_ms=(time.perf_counter() - started_total) * 1000.0,
        peak_vram_mb=peak_vram_mb,
        avg_vram_mb=avg_vram_mb,
        pages=reports,
        config={
            "dpi": options.dpi,
            "chunk_size": options.chunk_size,
            "batch_size": options.batch_size,
            "batch_max_pixels": options.batch_max_pixels,
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "sampling": options.sampling,
            "draft_mode": options.draft_mode,
            "verify_native_text": options.verify_native_text,
            "fallback_to_native": options.fallback_to_native,
            "prompt": options.prompt,
            "vlm_model": str(options.vlm_model) if options.vlm_model else None,
            "vlm_device": options.vlm_device,
            "vlm_dtype": options.vlm_dtype,
            "vlm_backend": options.vlm_backend,
            "layout_source": str(options.layout_json) if options.layout_json else options.layout_mode,
            "layout_device": options.layout_device,
            "layout_threshold": options.layout_threshold,
        },
        stats=_pdf_stats(reports),
    )
    write_json(ensure_dir(options.out_dir) / "report.json", report)
    return report


def recognize_prepared_pages(
    options: PipelineOptions,
    prepared_pages: list[PreparedPage],
    generator: BlockGenerator,
    *,
    started_total: float | None = None,
) -> list[PageReport]:
    if started_total is None:
        started_total = time.perf_counter()

    gen_options = GenerationOptions(
        chunk_size=options.chunk_size,
        batch_size=options.batch_size,
        batch_max_pixels=options.batch_max_pixels,
        max_tokens=options.max_tokens,
        temperature=options.temperature,
        sampling=options.sampling,
        verify_native_text=options.verify_native_text,
        fallback_to_native=options.fallback_to_native,
        prompt=options.prompt,
    )
    native_direct_generator = NativeDraftGenerator(getattr(generator, "tokenizer", None))

    page_blocks: list[list[PageDemoBlock]] = []
    pending_requests: list[tuple[int, int, Path, LayoutBlock, NativeTextCandidate | None]] = []
    reports: list[PageReport | None] = [None] * len(prepared_pages)
    native_direct_ms_by_page = [0.0] * len(prepared_pages)
    vlm_ms_by_page = [0.0] * len(prepared_pages)

    for page_index, prepared in enumerate(prepared_pages):
        blocks: list[PageDemoBlock] = []
        page_blocks.append(blocks)
        for index, block in enumerate(prepared.layout.blocks):
            candidate = (
                prepared.native_candidates[index]
                if options.draft_mode != "none" and index < len(prepared.native_candidates)
                else None
            )
            blocks.append(
                PageDemoBlock(
                    index=index + 1,
                    class_name=block.class_name,
                    label=block.label,
                    score=block.score,
                    bbox=block.bbox,
                    native_text_draft=candidate.text if candidate else None,
                    native_text_quality=candidate.quality if candidate else None,
                    native_text=None,
                    recognition=None,
                    recognition_error=None,
                    crop=None,
                )
            )
            if candidate is not None and candidate.quality.direct_accept and not gen_options.verify_native_text:
                try:
                    recognition = native_direct_generator.recognize(prepared.rendered.image_path, block, candidate, gen_options)
                except Exception as exc:  # pragma: no cover - defensive reporting
                    blocks[index].recognition_error = str(exc)
                    continue
                blocks[index].recognition = recognition
                blocks[index].native_text = recognition.text if recognition.backend.startswith("pdf-native-text") else None
                if recognition.backend.startswith("pdf-native-text"):
                    native_direct_ms_by_page[page_index] += recognition.ms
                else:
                    vlm_ms_by_page[page_index] += recognition.ms
                continue
            pending_requests.append((page_index, index, prepared.rendered.image_path, block, candidate))

    if pending_requests:
        try:
            recognitions = generator.recognize_many_with_paths(
                [(image_path, block, native_candidate) for _, _, image_path, block, native_candidate in pending_requests],
                gen_options,
            )
        except Exception as exc:  # pragma: no cover - defensive reporting
            for page_index, block_index, _image_path, _block, _candidate in pending_requests:
                page_blocks[page_index][block_index].recognition_error = str(exc)
        else:
            for (page_index, block_index, _image_path, _block, _candidate), recognition in zip(pending_requests, recognitions):
                page_blocks[page_index][block_index].recognition = recognition
                if recognition is None:
                    continue
                page_blocks[page_index][block_index].native_text = recognition.text if recognition.backend.startswith("pdf-native-text") else None
                if recognition.backend.startswith("pdf-native-text"):
                    native_direct_ms_by_page[page_index] += recognition.ms
                else:
                    vlm_ms_by_page[page_index] += recognition.ms

    for page_index, prepared in enumerate(prepared_pages):
        markdown_path = prepared.rendered.image_path.parent / "page.md"
        source = str(options.pdf or options.image or prepared.rendered.image_path)
        write_markdown(markdown_path, source, prepared.rendered.image_path, (prepared.rendered.width, prepared.rendered.height), page_blocks[page_index])
        reports[page_index] = PageReport(
            schema_version=1,
            source=source,
            image=str(prepared.rendered.image_path),
            image_size=[prepared.rendered.width, prepared.rendered.height],
            block_count=len(page_blocks[page_index]),
            layout_ms=prepared.layout_ms,
            native_text_ms=prepared.native_ms,
            native_direct_ms=native_direct_ms_by_page[page_index],
            vlm_ms=vlm_ms_by_page[page_index],
            recognition_ms=vlm_ms_by_page[page_index],
            total_ms=(time.perf_counter() - started_total) * 1000.0,
            artifacts=PageArtifacts(
                layout_json=str(prepared.layout_path),
                native_text_json=str(prepared.native_path),
                markdown=str(markdown_path),
                rendered_image=str(prepared.rendered.image_path),
                crops_dir=None,
            ),
            blocks=page_blocks[page_index],
            config={
                "page": prepared.page,
                "dpi": options.dpi,
                "chunk_size": options.chunk_size,
                "max_tokens": options.max_tokens,
                "temperature": options.temperature,
                "sampling": options.sampling,
                "draft_mode": options.draft_mode,
                "verify_native_text": options.verify_native_text,
                "fallback_to_native": options.fallback_to_native,
                "prompt": options.prompt,
                "vlm_model": str(options.vlm_model) if options.vlm_model else None,
                "vlm_device": options.vlm_device,
                "vlm_dtype": options.vlm_dtype,
                "vlm_backend": options.vlm_backend,
                "layout_source": str(options.layout_json) if options.layout_json else options.layout_mode,
                "layout_device": options.layout_device,
                "layout_threshold": options.layout_threshold,
                "generator": generator.__class__.__name__,
            },
            stats=_page_stats(page_blocks[page_index]),
        )
        write_json(prepared.rendered.image_path.parent / "report.json", reports[page_index])

    return [report for report in reports if report is not None]


def _build_generator(options: PipelineOptions) -> BlockGenerator:
    if options.generator is not None:
        return options.generator
    if options.vlm_model is not None:
        return load_transformers_vlm(
            options.vlm_model,
            device=options.vlm_device,
            dtype=options.vlm_dtype,
            trust_remote_code=options.trust_remote_code,
            backend=options.vlm_backend,
        )
    return NativeDraftGenerator()


def _page_options(options: PipelineOptions, page_index: int) -> PipelineOptions:
    return PipelineOptions(
        pdf=options.pdf,
        image=None,
        page=page_index,
        dpi=options.dpi,
        out_dir=options.out_dir / f"page_{page_index:04}",
        layout_json=options.layout_json,
        layout_mode=options.layout_mode,
        layout_device=options.layout_device,
        layout_threshold=options.layout_threshold,
        chunk_size=options.chunk_size,
        batch_size=options.batch_size,
        batch_max_pixels=options.batch_max_pixels,
        max_tokens=options.max_tokens,
        temperature=options.temperature,
        sampling=options.sampling,
        draft_mode=options.draft_mode,
        verify_native_text=options.verify_native_text,
        fallback_to_native=options.fallback_to_native,
        generator=options.generator,
        vlm_model=options.vlm_model,
        vlm_device=options.vlm_device,
        vlm_dtype=options.vlm_dtype,
        vlm_backend=options.vlm_backend,
        trust_remote_code=options.trust_remote_code,
        prompt=options.prompt,
    )


def prepare_page(options: PipelineOptions) -> PreparedPage:
    started_total = time.perf_counter()
    out_dir = ensure_dir(options.out_dir)
    rendered = _render_or_copy_input(options, out_dir)
    layout_path = out_dir / "layout.json"
    detector = _build_layout_detector(options, rendered)
    layout, layout_ms = detect_layout(detector, rendered.image_path, (rendered.width, rendered.height), layout_path)
    native_path = out_dir / "native_text.json"
    native_candidates, native_ms = _extract_native_candidates(options, rendered, layout.blocks, native_path)
    _ = started_total
    return PreparedPage(
        page=options.page,
        rendered=rendered,
        layout=layout,
        layout_ms=layout_ms,
        native_candidates=native_candidates,
        native_ms=native_ms,
        layout_path=layout_path,
        native_path=native_path,
    )


def recognize_prepared_page(
    options: PipelineOptions,
    prepared: PreparedPage,
    generator: BlockGenerator,
    *,
    started_total: float | None = None,
) -> PageReport:
    if started_total is None:
        started_total = time.perf_counter()
    gen_options = GenerationOptions(
        chunk_size=options.chunk_size,
        batch_size=options.batch_size,
        batch_max_pixels=options.batch_max_pixels,
        max_tokens=options.max_tokens,
        temperature=options.temperature,
        sampling=options.sampling,
        verify_native_text=options.verify_native_text,
        fallback_to_native=options.fallback_to_native,
        prompt=options.prompt,
    )
    native_direct_generator = NativeDraftGenerator(getattr(generator, "tokenizer", None))
    page_blocks: list[PageDemoBlock] = []
    native_direct_ms = 0.0
    vlm_ms = 0.0
    pending_requests: list[tuple[int, LayoutBlock, NativeTextCandidate | None]] = []
    for index, block in enumerate(prepared.layout.blocks):
        candidate = (
            prepared.native_candidates[index]
            if options.draft_mode != "none" and index < len(prepared.native_candidates)
            else None
        )
        page_blocks.append(
            PageDemoBlock(
                index=index + 1,
                class_name=block.class_name,
                label=block.label,
                score=block.score,
                bbox=block.bbox,
                native_text_draft=candidate.text if candidate else None,
                native_text_quality=candidate.quality if candidate else None,
                native_text=None,
                recognition=None,
                recognition_error=None,
                crop=None,
            )
        )

        if candidate is not None and candidate.quality.direct_accept and not gen_options.verify_native_text:
            try:
                recognition = native_direct_generator.recognize(prepared.rendered.image_path, block, candidate, gen_options)
            except Exception as exc:  # pragma: no cover - defensive reporting
                page_blocks[index].recognition_error = str(exc)
                continue
            page_blocks[index].recognition = recognition
            page_blocks[index].native_text = recognition.text if recognition.backend.startswith("pdf-native-text") else None
            if recognition.backend.startswith("pdf-native-text"):
                native_direct_ms += recognition.ms
            else:
                vlm_ms += recognition.ms
            continue

        pending_requests.append((index, block, candidate))

    if pending_requests:
        try:
            recognitions = generator.recognize_many(
                prepared.rendered.image_path,
                [(block, native_candidate) for _, block, native_candidate in pending_requests],
                gen_options,
            )
        except Exception as exc:  # pragma: no cover - defensive reporting
            for index, _, _ in pending_requests:
                page_blocks[index].recognition_error = str(exc)
        else:
            for (index, _, _), recognition in zip(pending_requests, recognitions):
                page_blocks[index].recognition = recognition
                if recognition is None:
                    continue
                page_blocks[index].native_text = recognition.text if recognition.backend.startswith("pdf-native-text") else None
                if recognition.backend.startswith("pdf-native-text"):
                    native_direct_ms += recognition.ms
                else:
                    vlm_ms += recognition.ms
    markdown_path = prepared.rendered.image_path.parent / "page.md"
    source = str(options.pdf or options.image or prepared.rendered.image_path)
    write_markdown(markdown_path, source, prepared.rendered.image_path, (prepared.rendered.width, prepared.rendered.height), page_blocks)
    report = PageReport(
        schema_version=1,
        source=source,
        image=str(prepared.rendered.image_path),
        image_size=[prepared.rendered.width, prepared.rendered.height],
        block_count=len(page_blocks),
        layout_ms=prepared.layout_ms,
        native_text_ms=prepared.native_ms,
        native_direct_ms=native_direct_ms,
        vlm_ms=vlm_ms,
        recognition_ms=vlm_ms,
        total_ms=(time.perf_counter() - started_total) * 1000.0,
        artifacts=PageArtifacts(
            layout_json=str(prepared.layout_path),
            native_text_json=str(prepared.native_path),
            markdown=str(markdown_path),
            rendered_image=str(prepared.rendered.image_path),
            crops_dir=None,
        ),
        blocks=page_blocks,
        config={
            "page": options.page,
            "dpi": options.dpi,
            "chunk_size": options.chunk_size,
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "sampling": options.sampling,
            "draft_mode": options.draft_mode,
            "verify_native_text": options.verify_native_text,
            "fallback_to_native": options.fallback_to_native,
            "prompt": options.prompt,
            "vlm_model": str(options.vlm_model) if options.vlm_model else None,
            "vlm_device": options.vlm_device,
            "vlm_dtype": options.vlm_dtype,
            "vlm_backend": options.vlm_backend,
            "layout_source": str(options.layout_json) if options.layout_json else options.layout_mode,
            "layout_device": options.layout_device,
            "layout_threshold": options.layout_threshold,
            "generator": generator.__class__.__name__,
        },
        stats=_page_stats(page_blocks),
    )
    write_json(prepared.rendered.image_path.parent / "report.json", report)
    return report


def _extract_native_candidates(
    options: PipelineOptions,
    rendered: RenderedPage,
    blocks: list[DetectionResult | object],
    native_path: Path,
) -> tuple[list[NativeTextCandidate | None], float]:
    started_native = time.perf_counter()
    if options.pdf:
        native_candidates = extract_native_text_candidates_for_blocks(
            options.pdf,
            options.page,
            options.dpi,
            rendered.width,
            rendered.height,
            list(blocks),  # type: ignore[arg-type]
            native_path,
        )
    else:
        native_candidates = []
        write_json(
            native_path,
            {
                "schema_version": 1,
                "pdf": None,
                "page": options.page,
                "dpi": options.dpi,
                "scale": rendered.scale,
                "image_size": [rendered.width, rendered.height],
                "blocks": [],
            },
        )
    native_ms = (time.perf_counter() - started_native) * 1000.0
    return native_candidates, native_ms


def _build_layout_detector(options: PipelineOptions, rendered: RenderedPage):
    if options.layout_json:
        return JsonLayoutDetector(options.layout_json)
    if options.layout_mode == "pdf-lines":
        if options.pdf is None:
            raise ValueError("--layout-mode pdf-lines requires --pdf input")
        blocks = detect_pdf_text_line_blocks(
            options.pdf,
            options.page,
            options.dpi,
            rendered.width,
            rendered.height,
        )
        return ProvidedBlocksLayoutDetector(blocks, model_id="pdf-native-line-layout")
    if options.layout_mode == "pp-doclayout-v2":
        return PaddleOCRLayoutDetector(
            model_name="PP-DocLayoutV2",
            device=options.layout_device,
            threshold=options.layout_threshold,
        )
    return WholePageLayoutDetector()


def _render_or_copy_input(options: PipelineOptions, out_dir: Path) -> RenderedPage:
    if options.pdf is not None:
        image_path = out_dir / f"page_{options.page:04}.png"
        return render_pdf_page_to_png(options.pdf, options.page, options.dpi, image_path)
    if options.image is not None:
        width, height = _image_size(options.image)
        image_path = out_dir / options.image.name
        if options.image.resolve() != image_path.resolve():
            shutil.copyfile(options.image, image_path)
        return RenderedPage(
            image_path=image_path,
            width=width,
            height=height,
            scale=1.0,
            source=str(options.image),
        )
    raise ValueError("either pdf or image must be provided")


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Pillow is required for image inputs; install the project with uv sync") from exc
    with Image.open(path) as image:
        return image.size


def _page_stats(blocks: list[PageDemoBlock]) -> dict[str, float | int]:
    total_blocks = len(blocks)
    native_eligible_blocks = 0
    draft_candidate_blocks = 0
    direct_quality_blocks = 0
    recognized_blocks = 0
    fallback_blocks = 0
    draft_tokens = 0
    accepted_tokens = 0
    matched_tokens = 0
    rejected_tokens = 0
    rollback_tokens = 0
    generated_tokens = 0
    direct_accept_blocks = 0
    native_direct_accept_blocks = 0
    verified_accept_blocks = 0
    prefix_accept_blocks = 0
    prefix_lengths: list[int] = []
    for block in blocks:
        if should_use_native_text(block.class_name):
            native_eligible_blocks += 1
        if block.native_text_draft:
            draft_candidate_blocks += 1
        if block.native_text_quality and block.native_text_quality.direct_accept:
            direct_quality_blocks += 1
        if block.recognition is not None:
            recognized_blocks += 1
            if "fallback" in block.recognition.backend:
                fallback_blocks += 1
        if block.recognition is None or block.recognition.draft is None:
            continue
        stats = block.recognition.draft
        draft_tokens += stats.draft_tokens
        accepted_tokens += stats.accepted_tokens
        matched_tokens += stats.matched_tokens
        rejected_tokens += stats.rejected_tokens
        rollback_tokens += stats.rollback_tokens
        generated_tokens += stats.generated_tokens
        if stats.accepted:
            direct_accept_blocks += 1
            if stats.mode == "direct_accept" or block.recognition.backend == "pdf-native-text":
                native_direct_accept_blocks += 1
            else:
                verified_accept_blocks += 1
        elif stats.prefix_accepted:
            prefix_accept_blocks += 1
        if stats.prefix_accepted:
            prefix_lengths.append(stats.accepted_tokens)
    ratio = accepted_tokens / draft_tokens if draft_tokens else 0.0
    coverage = draft_candidate_blocks / native_eligible_blocks if native_eligible_blocks else 0.0
    hit_rate = direct_accept_blocks / draft_candidate_blocks if draft_candidate_blocks else 0.0
    rollback_ratio = rollback_tokens / draft_tokens if draft_tokens else 0.0
    return {
        "total_blocks": total_blocks,
        "native_eligible_blocks": native_eligible_blocks,
        "draft_candidate_blocks": draft_candidate_blocks,
        "draft_coverage": coverage,
        "direct_quality_blocks": direct_quality_blocks,
        "recognized_blocks": recognized_blocks,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "matched_tokens": matched_tokens,
        "rejected_tokens": rejected_tokens,
        "generated_tokens": generated_tokens,
        "rollback_tokens": rollback_tokens,
        "accepted_token_ratio": ratio,
        "rollback_ratio": rollback_ratio,
        "draft_hit_rate": hit_rate,
        "average_prefix_accepted_tokens": (
            sum(prefix_lengths) / len(prefix_lengths) if prefix_lengths else 0.0
        ),
        "direct_accept_blocks": direct_accept_blocks,
        "native_direct_accept_blocks": native_direct_accept_blocks,
        "verified_accept_blocks": verified_accept_blocks,
        "prefix_accept_blocks": prefix_accept_blocks,
        "fallback_blocks": fallback_blocks,
    }


def _pdf_stats(reports: list[PageReport]) -> dict[str, float | int]:
    total_blocks = sum(report.block_count for report in reports)
    total_layout_ms = sum(report.layout_ms for report in reports)
    total_native_text_ms = sum(report.native_text_ms for report in reports)
    total_native_direct_ms = sum(report.native_direct_ms for report in reports)
    total_vlm_ms = sum(report.vlm_ms for report in reports)
    total_accepted_tokens = sum(report.stats.get("accepted_tokens", 0) for report in reports)
    total_draft_tokens = sum(report.stats.get("draft_tokens", 0) for report in reports)
    total_direct_accept_blocks = sum(report.stats.get("direct_accept_blocks", 0) for report in reports)
    total_verified_accept_blocks = sum(report.stats.get("verified_accept_blocks", 0) for report in reports)
    total_prefix_accept_blocks = sum(report.stats.get("prefix_accept_blocks", 0) for report in reports)
    return {
        "page_count": len(reports),
        "block_count": total_blocks,
        "total_layout_ms": total_layout_ms,
        "total_native_text_ms": total_native_text_ms,
        "total_native_direct_ms": total_native_direct_ms,
        "total_vlm_ms": total_vlm_ms,
        "accepted_tokens": total_accepted_tokens,
        "draft_tokens": total_draft_tokens,
        "accepted_token_ratio": total_accepted_tokens / total_draft_tokens if total_draft_tokens else 0.0,
        "direct_accept_blocks": total_direct_accept_blocks,
        "verified_accept_blocks": total_verified_accept_blocks,
        "prefix_accept_blocks": total_prefix_accept_blocks,
    }


def _cuda_memory_stats() -> tuple[float | None, float | None]:
    try:
        import torch
    except Exception:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    try:
        peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        current = float(torch.cuda.memory_allocated() / (1024 * 1024))
        return peak, current
    except Exception:
        return None, None


def _has_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _reset_cuda_peak_memory_stats() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
