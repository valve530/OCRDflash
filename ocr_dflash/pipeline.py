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
    PageArtifacts,
    PageDemoBlock,
    PageReport,
    RenderedPage,
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


def run_page_pipeline(options: PipelineOptions) -> PageReport:
    started_total = time.perf_counter()
    out_dir = ensure_dir(options.out_dir)
    rendered = _render_or_copy_input(options, out_dir)

    layout_path = out_dir / "layout.json"
    detector = _build_layout_detector(options, rendered)
    layout, layout_ms = detect_layout(detector, rendered.image_path, (rendered.width, rendered.height), layout_path)

    native_path = out_dir / "native_text.json"
    started_native = time.perf_counter()
    native_candidates: list[NativeTextCandidate | None]
    if options.pdf:
        native_candidates = extract_native_text_candidates_for_blocks(
            options.pdf,
            options.page,
            options.dpi,
            rendered.width,
            rendered.height,
            layout.blocks,
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

    gen_options = GenerationOptions(
        chunk_size=options.chunk_size,
        max_tokens=options.max_tokens,
        temperature=options.temperature,
        sampling=options.sampling,
        verify_native_text=options.verify_native_text,
        fallback_to_native=options.fallback_to_native,
        prompt=options.prompt,
    )
    generator = _build_generator(options)
    started_recognition = time.perf_counter()
    page_blocks: list[PageDemoBlock] = []
    for index, block in enumerate(layout.blocks):
        candidate = (
            native_candidates[index]
            if options.draft_mode != "none" and index < len(native_candidates)
            else None
        )
        recognition: BlockRecognition | None = None
        error: str | None = None
        try:
            recognition = generator.recognize(rendered.image_path, block, candidate, gen_options)
        except Exception as exc:  # pragma: no cover - defensive reporting
            error = str(exc)
        native_text = recognition.text if recognition and recognition.backend.startswith("pdf-native-text") else None
        page_blocks.append(
            PageDemoBlock(
                index=index + 1,
                class_name=block.class_name,
                label=block.label,
                score=block.score,
                bbox=block.bbox,
                native_text_draft=candidate.text if candidate else None,
                native_text_quality=candidate.quality if candidate else None,
                native_text=native_text,
                recognition=recognition,
                recognition_error=error,
                crop=None,
            )
        )
    recognition_ms = (time.perf_counter() - started_recognition) * 1000.0

    markdown_path = out_dir / "page.md"
    source = str(options.pdf or options.image or rendered.image_path)
    write_markdown(markdown_path, source, rendered.image_path, (rendered.width, rendered.height), page_blocks)

    report = PageReport(
        schema_version=1,
        source=source,
        image=str(rendered.image_path),
        image_size=[rendered.width, rendered.height],
        block_count=len(page_blocks),
        layout_ms=layout_ms,
        native_text_ms=native_ms,
        recognition_ms=recognition_ms,
        total_ms=(time.perf_counter() - started_total) * 1000.0,
        artifacts=PageArtifacts(
            layout_json=str(layout_path),
            native_text_json=str(native_path),
            markdown=str(markdown_path),
            rendered_image=str(rendered.image_path),
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
    write_json(out_dir / "report.json", report)
    return report


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
        "prefix_accept_blocks": prefix_accept_blocks,
        "fallback_blocks": fallback_blocks,
    }
