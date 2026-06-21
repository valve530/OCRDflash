from __future__ import annotations

import argparse
from pathlib import Path

from .eval import compare_report_blocks, compare_text_files, summarize_dflash_report, write_text_metrics
from .pipeline import PipelineOptions, run_page_pipeline, run_pdf_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-dflash",
        description="PDF-native draft-guided document OCR research pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse-page", help="render/extract/recognize one PDF page or image")
    source = parse.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path)
    source.add_argument("--image", type=Path)
    parse.add_argument("--page", type=int, default=0)
    parse.add_argument("--dpi", type=int, default=200)
    parse.add_argument("--out-dir", type=Path, required=True)
    parse.add_argument("--layout-json", type=Path)
    parse.add_argument(
        "--layout-mode",
        choices=["whole-page", "pdf-lines", "pp-doclayout-v2"],
        default="whole-page",
        help="whole-page uses one block; pdf-lines chunks PDF text lines/paragraphs; pp-doclayout-v2 uses PaddleOCR layout",
    )
    parse.add_argument("--layout-device")
    parse.add_argument("--layout-threshold", type=float, default=0.5)
    parse.add_argument("--chunk-size", type=int, default=16)
    parse.add_argument("--batch-size", type=int, default=128)
    parse.add_argument("--batch-max-pixels", type=int, default=0)
    parse.add_argument("--max-tokens", type=int, default=256)
    parse.add_argument("--temperature", type=float, default=0.0)
    parse.add_argument("--sampling", action="store_true")
    parse.add_argument(
        "--draft-mode",
        choices=["native", "none"],
        default="native",
        help="native uses PDF text drafts; none disables drafts for ablation",
    )
    parse.add_argument("--verify-native-text", action="store_true")
    parse.add_argument("--no-native-fallback", action="store_true")
    parse.add_argument("--vlm-model")
    parse.add_argument("--vlm-backend", choices=["auto", "transformers", "paddleocr-vl"], default="auto")
    parse.add_argument("--vlm-device", default="auto")
    parse.add_argument("--vlm-dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parse.add_argument("--vlm-attn-implementation", default=None)
    parse.add_argument("--vlm-max-pixels", type=int, default=None)
    parse.add_argument("--no-trust-remote-code", action="store_true")
    parse.add_argument("--prompt", default="Convert this document image region to Markdown.")

    parse_pdf = sub.add_parser("parse-pdf", help="process all pages in one PDF with one model load")
    parse_pdf.add_argument("--pdf", type=Path, required=True)
    parse_pdf.add_argument("--out-dir", type=Path, required=True)
    parse_pdf.add_argument("--dpi", type=int, default=200)
    parse_pdf.add_argument("--layout-json", type=Path)
    parse_pdf.add_argument(
        "--layout-mode",
        choices=["whole-page", "pdf-lines", "pp-doclayout-v2"],
        default="whole-page",
    )
    parse_pdf.add_argument("--layout-device")
    parse_pdf.add_argument("--layout-threshold", type=float, default=0.5)
    parse_pdf.add_argument("--chunk-size", type=int, default=16)
    parse_pdf.add_argument("--batch-size", type=int, default=128)
    parse_pdf.add_argument("--batch-max-pixels", type=int, default=0)
    parse_pdf.add_argument("--max-tokens", type=int, default=256)
    parse_pdf.add_argument("--temperature", type=float, default=0.0)
    parse_pdf.add_argument("--sampling", action="store_true")
    parse_pdf.add_argument(
        "--draft-mode",
        choices=["native", "none"],
        default="native",
    )
    parse_pdf.add_argument("--verify-native-text", action="store_true")
    parse_pdf.add_argument("--no-native-fallback", action="store_true")
    parse_pdf.add_argument("--vlm-model")
    parse_pdf.add_argument("--vlm-backend", choices=["auto", "transformers", "paddleocr-vl"], default="auto")
    parse_pdf.add_argument("--vlm-device", default="auto")
    parse_pdf.add_argument("--vlm-dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parse_pdf.add_argument("--vlm-attn-implementation", default=None)
    parse_pdf.add_argument("--vlm-max-pixels", type=int, default=None)
    parse_pdf.add_argument("--no-trust-remote-code", action="store_true")
    parse_pdf.add_argument("--prompt", default="Convert this document image region to Markdown.")

    compare = sub.add_parser("compare-text", help="compute quality metrics for text or reports")
    compare.add_argument("--expected", type=Path, required=True)
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--out", type=Path)
    compare.add_argument(
        "--kind",
        choices=["text", "report"],
        default="text",
        help="text compares raw files; report compares block texts in report.json",
    )

    analyze = sub.add_parser("analyze-report", help="summarize DFlash acceptance from report.json")
    analyze.add_argument("report", type=Path)
    analyze.add_argument("--prefix", type=int, default=120)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "parse-page":
        report = run_page_pipeline(
            PipelineOptions(
                pdf=args.pdf,
                image=args.image,
                page=args.page,
                dpi=args.dpi,
                out_dir=args.out_dir,
                layout_json=args.layout_json,
                layout_mode=args.layout_mode,
                layout_device=args.layout_device,
                layout_threshold=args.layout_threshold,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
                batch_max_pixels=args.batch_max_pixels,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                sampling=args.sampling,
                draft_mode=args.draft_mode,
                verify_native_text=args.verify_native_text,
                fallback_to_native=not args.no_native_fallback,
                vlm_model=args.vlm_model,
                vlm_backend=args.vlm_backend,
                vlm_device=args.vlm_device,
                vlm_dtype=args.vlm_dtype,
                vlm_attn_implementation=args.vlm_attn_implementation,
                vlm_max_pixels=args.vlm_max_pixels,
                trust_remote_code=not args.no_trust_remote_code,
                prompt=args.prompt,
            )
        )
        print(f"wrote {report.artifacts.markdown}")
        print(f"wrote {args.out_dir / 'report.json'}")
    elif args.command == "parse-pdf":
        report = run_pdf_pipeline(
            PipelineOptions(
                pdf=args.pdf,
                out_dir=args.out_dir,
                dpi=args.dpi,
                layout_json=args.layout_json,
                layout_mode=args.layout_mode,
                layout_device=args.layout_device,
                layout_threshold=args.layout_threshold,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
                batch_max_pixels=args.batch_max_pixels,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                sampling=args.sampling,
                draft_mode=args.draft_mode,
                verify_native_text=args.verify_native_text,
                fallback_to_native=not args.no_native_fallback,
                vlm_model=args.vlm_model,
                vlm_backend=args.vlm_backend,
                vlm_device=args.vlm_device,
                vlm_dtype=args.vlm_dtype,
                vlm_attn_implementation=args.vlm_attn_implementation,
                vlm_max_pixels=args.vlm_max_pixels,
                trust_remote_code=not args.no_trust_remote_code,
                prompt=args.prompt,
            )
        )
        print(f"wrote {args.out_dir / 'report.json'}")
    elif args.command == "compare-text":
        metrics = (
            compare_text_files(args.expected, args.actual)
            if args.kind == "text"
            else compare_report_blocks(args.expected, args.actual)
        )
        if args.out:
            write_text_metrics(args.out, metrics)
            print(f"wrote {args.out}")
        else:
            import json
            from .schemas import to_jsonable

            print(json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2))
    elif args.command == "analyze-report":
        print(summarize_dflash_report(args.report, prefix=args.prefix))
    else:  # pragma: no cover
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
