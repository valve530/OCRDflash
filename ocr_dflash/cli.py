from __future__ import annotations

import argparse
import os
from pathlib import Path

from .simple_pipeline import SimpleOptions, copy_pdf_outputs, run_pdf


DEFAULT_PROXY = "http://100.64.0.250:7890"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocr-dflash")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/ocr_dflash_out"))
    parser.add_argument("--page", type=int, default=None)
    parser.add_argument("--mode", choices=["dflash", "baseline"], default="dflash")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--layout-model-dir", type=Path, default=Path("models/PP-DocLayoutV3"))
    parser.add_argument("--vlm-model", type=Path, default=Path("models/PaddleOCR-VL-1.6"))
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--table-native-draft", action="store_true")
    parser.add_argument("--proxy", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_env(args.proxy)
    report = run_pdf(
        SimpleOptions(
            pdf=args.pdf,
            out_dir=args.out_dir,
            page=args.page,
            dpi=args.dpi,
            layout_model_dir=args.layout_model_dir,
            vlm_model=args.vlm_model,
            mode=args.mode,
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
            max_pixels=args.max_pixels,
            table_native_draft=args.table_native_draft,
        )
    )
    copy_pdf_outputs(report, args.out_dir)
    print(f"report: {args.out_dir / 'report.json'}")
    print(f"pages: {report.page_count}")
    print(f"layout_ms: {report.total_layout_ms:.1f}")
    print(f"native_ms: {report.total_native_text_ms:.1f}")
    print(f"vlm_ms: {report.total_vlm_ms:.1f}")
    print(f"accepted: {report.stats.get('accepted_tokens', 0)}/{report.stats.get('draft_tokens', 0)}")
    print(f"accepted_ratio: {report.stats.get('accepted_token_ratio', 0.0):.2%}")
    if report.peak_vram_mb is not None:
        print(f"peak_vram_mb: {report.peak_vram_mb:.1f}")


def setup_env(proxy: str | None) -> None:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if proxy is None:
        return
    value = DEFAULT_PROXY if proxy == "default" else proxy
    os.environ.setdefault("HTTP_PROXY", value)
    os.environ.setdefault("HTTPS_PROXY", value)
    os.environ.setdefault("ALL_PROXY", value)


if __name__ == "__main__":
    main()
