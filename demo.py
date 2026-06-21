from __future__ import annotations

import argparse
import os
from pathlib import Path

from ocr_dflash.pipeline import PipelineOptions, run_page_pipeline


DEFAULT_PROXY = "http://100.64.0.250:7890"


def main() -> None:
    args = _parse_args()
    _setup_env(args.proxy)

    report = run_page_pipeline(
        PipelineOptions(
            pdf=args.pdf,
            page=args.page,
            dpi=args.dpi,
            out_dir=args.out_dir,
            layout_mode=args.layout_mode,
            layout_device=args.layout_device,
            layout_threshold=args.layout_threshold,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            batch_max_pixels=args.batch_max_pixels,
            max_tokens=args.max_tokens,
            sampling=args.sampling,
            verify_native_text=args.verify_native_text,
            fallback_to_native=not args.no_native_fallback,
            vlm_model=args.vlm_model,
            vlm_device=args.vlm_device,
            vlm_dtype=args.vlm_dtype,
            vlm_backend=args.vlm_backend,
            vlm_attn_implementation=args.vlm_attn_implementation,
            vlm_max_pixels=args.vlm_max_pixels,
            trust_remote_code=not args.no_trust_remote_code,
            prompt=args.prompt,
        )
    )

    print(f"markdown: {report.artifacts.markdown}")
    print(f"report:   {args.out_dir / 'report.json'}")
    print(f"layout:   {report.artifacts.layout_json}")
    print(f"blocks:   {report.block_count}")
    print(f"stats:    {report.stats}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug demo for PP-DocLayoutV2 + PaddleOCR-VL-1.6.")
    parser.add_argument("--pdf", type=Path, default=Path("tmp/attention_1.pdf"))
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/demo_vlm"))
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--layout-mode", default="pp-doclayout-v2")
    parser.add_argument("--layout-device", default="cpu")
    parser.add_argument("--layout-threshold", type=float, default=0.3)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-max-pixels", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--verify-native-text", action="store_true")
    parser.add_argument("--no-native-fallback", action="store_true")
    parser.add_argument("--vlm-model", default="./models/PaddleOCR-VL-1.6")
    parser.add_argument("--vlm-backend", default="paddleocr-vl")
    parser.add_argument("--vlm-device", default="auto")
    parser.add_argument("--vlm-dtype", default="bf16")
    parser.add_argument("--vlm-attn-implementation", default="flash_attention_2")
    parser.add_argument("--vlm-max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--prompt", default="Convert this document image region to Markdown.")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    return parser.parse_args()


def _setup_env(proxy: str | None) -> None:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if proxy:
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)
        os.environ.setdefault("ALL_PROXY", proxy)


if __name__ == "__main__":
    main()
