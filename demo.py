from __future__ import annotations

from pathlib import Path

from ocr_dflash.simple_pipeline import SimpleOptions, copy_pdf_outputs, run_pdf


def main() -> None:
    out_dir = Path("tmp/demo_simple")
    report = run_pdf(
        SimpleOptions(
            pdf=Path("tmp/attention_1.pdf"),
            out_dir=out_dir,
            mode="dflash",
            max_tokens=128,
            chunk_size=8,
        )
    )
    copy_pdf_outputs(report, out_dir)
    print(f"report: {out_dir / 'report.json'}")
    print(f"accepted: {report.stats.get('accepted_tokens', 0)}/{report.stats.get('draft_tokens', 0)}")
    print(f"vlm_ms: {report.total_vlm_ms:.1f}")


if __name__ == "__main__":
    main()
