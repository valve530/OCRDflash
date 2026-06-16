# ocr-dflash

Python research implementation of PDF-native draft-guided document OCR.

The pipeline follows `docs/python_pdf_vlm_acceleration_prd.md` and mirrors the
artifact shape used by `/home/valve/flash-doc-rs`:

- `layout.json`
- `native_text.json`
- `page.md`
- `report.json`

## Install

```sh
uv sync --dev
```

If the network is restricted, use the local proxy:

```sh
HTTP_PROXY=http://100.64.0.250:7890 \
HTTPS_PROXY=http://100.64.0.250:7890 \
ALL_PROXY=http://100.64.0.250:7890 \
UV_CACHE_DIR=/tmp/uv-cache \
uv sync --dev
```

## Parse One Page

```sh
uv run ocr-dflash parse-page \
  --pdf sample.pdf \
  --page 0 \
  --dpi 200 \
  --out-dir out/sample
```

For image input:

```sh
uv run ocr-dflash parse-page \
  --image page.png \
  --out-dir out/page
```

An external layout result can be supplied with `--layout-json`; otherwise the
research baseline uses one whole-page text block. For PDFs, `--layout-mode
pdf-lines` builds block crops from the PDF text line boxes so native drafts can
be verified block by block. If the optional PaddleOCR layout package/model is
available, `--layout-mode pp-doclayout-v2` runs PP-DocLayoutV2 and writes the
same `layout.json` schema.

Useful research switches:

```sh
# Disable PDF drafts for ablation.
uv run ocr-dflash parse-page \
  --pdf sample.pdf \
  --out-dir out/no-draft \
  --draft-mode none

# Force native drafts through the verification path once a verifier adapter is
# plugged in; without a concrete VLM adapter this disables direct accept.
uv run ocr-dflash parse-page \
  --pdf sample.pdf \
  --out-dir out/verify \
  --verify-native-text

# Change chunking and decoding controls recorded in report.json.
uv run ocr-dflash parse-page \
  --pdf sample.pdf \
  --out-dir out/chunk8 \
  --chunk-size 8 \
  --max-tokens 512 \
  --sampling \
  --temperature 0.7
```

`report.json` records draft coverage, accepted token ratio, rollback ratio,
average accepted prefix length, direct/prefix/fallback block counts, and the
generation config used for the run.

To use a Hugging Face / PaddleOCR-VL style model for block generation:

```sh
HTTP_PROXY=http://100.64.0.250:7890 \
HTTPS_PROXY=http://100.64.0.250:7890 \
ALL_PROXY=http://100.64.0.250:7890 \
HF_HUB_DISABLE_XET=1 \
uv run ocr-dflash parse-page \
  --pdf sample.pdf \
  --out-dir out/vlm \
  --layout-mode pdf-lines \
  --vlm-model PaddlePaddle/PaddleOCR-VL \
  --vlm-backend paddleocr-vl \
  --vlm-device auto \
  --vlm-dtype bf16 \
  --verify-native-text \
  --chunk-size 8 \
  --max-tokens 256
```

Without `--vlm-model`, `parse-page` deliberately stays on the fast PDF native
text baseline and no model is loaded. With `--vlm-backend paddleocr-vl`, the
pipeline loads `PaddleOCRVLForConditionalGeneration`, formats the image prompt
with the PaddleOCR-VL chat template, verifies PDF native text draft tokens, and
falls back to VLM generation after the first mismatch.

The generic adapter uses `AutoProcessor` and `AutoModelForImageTextToText`
first, then falls back to `AutoModelForCausalLM`.

For a real layout model instead of PDF text-line chunks:

```sh
uv run ocr-dflash parse-page \
  --pdf sample.pdf \
  --out-dir out/layout-vlm \
  --layout-mode pp-doclayout-v2 \
  --layout-device gpu \
  --vlm-model PaddlePaddle/PaddleOCR-VL \
  --vlm-backend paddleocr-vl
```

`pp-doclayout-v2` is optional because PaddleOCR/Paddle wheels are heavy and
platform-specific. Without that package installed, use `pdf-lines` or
`--layout-json`.

## Evaluation Helpers

```sh
uv run ocr-dflash compare-text \
  --expected baseline/page.md \
  --actual experiment/page.md

uv run ocr-dflash compare-text \
  --kind report \
  --expected baseline/report.json \
  --actual experiment/report.json \
  --out compare.json
```

The helper reports character accuracy, edit distance, exact match, and block
exact-match ratios for `report.json` files.

## Current Model Boundary

The implemented P0/P1 research scaffold includes:

- PDF page rendering with PyMuPDF
- PDF native text extraction and Rust-aligned quality heuristics
- layout adapter layer with whole-page, PDF-line, external JSON, and optional
  PaddleOCR PP-DocLayoutV2 modes
- native draft direct accept / fallback behavior
- token-level draft verification primitives
- PaddleOCR-VL block generation with draft verification / prefix continuation
- block/page reports with accepted, matched, generated, and rollback stats
- Markdown assembly

`DraftVerifyingGenerator` remains the extension point for another document VLM.
`PaddleOCRVLDFlashGenerator` is the concrete PaddleOCR-VL adapter. Its current
Python implementation is intentionally correctness-first: it re-runs a forward
pass for draft-token checks instead of using the Rust/MLX KV-cache optimized
loop.

## Tests

```sh
uv run pytest -q
```
