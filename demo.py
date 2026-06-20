from __future__ import annotations

import argparse
import os
from pathlib import Path

from ocr_dflash.pipeline import PipelineOptions, run_page_pipeline


DEFAULT_PROXY = "http://100.64.0.250:7890"


def main() -> None:
    args = _parse_args()
    _setup_debug_env(args.proxy)

    options = PipelineOptions(
        pdf=args.pdf,
        page=args.page,
        dpi=args.dpi,
        out_dir=args.out_dir,
        layout_json=args.layout_json,
        layout_mode="pp-doclayout-v2" if args.layout_json is None else "whole-page",
        layout_threshold=args.layout_threshold,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        vlm_model=args.vlm_model,
        vlm_backend="paddleocr-vl",
        vlm_device=args.vlm_device,
        vlm_dtype=args.vlm_dtype,
    )

    report = run_page_pipeline(options)
    print(f"markdown: {report.artifacts.markdown}")
    print(f"report:   {args.out_dir / 'report.json'}")
    print(f"layout:   {report.artifacts.layout_json}")
    print(f"blocks:   {report.block_count}")
    print(f"stats:    {report.stats}")
    print("backends:")
    for backend in sorted(
        {
            block.recognition.backend
            for block in report.blocks
            if block.recognition is not None
        }
    ):
        print(f"  - {backend}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug demo for PP-DocLayout + PaddleOCR-VL DFlash OCR.")
    parser.add_argument("--pdf", type=Path, default=Path("tmp/attention_1.pdf"))
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/demo_vlm"))
    parser.add_argument("--layout-json", type=Path)
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--layout-threshold", type=float, default=0.3)
    parser.add_argument("--vlm-model", default="PaddlePaddle/PaddleOCR-VL")
    parser.add_argument("--vlm-device", default="auto")
    parser.add_argument("--vlm-dtype", default="bf16")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    return parser.parse_args()


def _setup_debug_env(proxy: str | None) -> None:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if proxy:
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)
        os.environ.setdefault("ALL_PROXY", proxy)


if __name__ == "__main__":
    main()
