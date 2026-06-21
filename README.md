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
  --vlm-model ./models/PaddleOCR-VL-1.6 \
  --vlm-backend paddleocr-vl \
  --vlm-device auto \
  --vlm-dtype bf16 \
  --vlm-attn-implementation flash_attention_2 \
  --vlm-max-pixels 1003520 \
  --verify-native-text \
  --chunk-size 8 \
  --max-tokens 256
```

If the model is not already present in the Hugging Face cache, download it
first. With a restricted network, use the local proxy and disable Xet:

```sh
HTTP_PROXY=http://100.64.0.250:7890 \
HTTPS_PROXY=http://100.64.0.250:7890 \
ALL_PROXY=http://100.64.0.250:7890 \
HF_HUB_DISABLE_XET=1 \
uv run hf download PaddlePaddle/PaddleOCR-VL-1.6 \
  --include "model.safetensors" \
  --include "*.json" \
  --include "*.py" \
  --include "*.jinja" \
  --include "tokenizer.*" \
  --include "preprocessor_config.json"
```

To keep the model inside the project instead of the global Hugging Face cache,
add `--local-dir ./models/PaddleOCR-VL-1.6` and pass
`--vlm-model ./models/PaddleOCR-VL-1.6` when running `parse-page`.

Without `--vlm-model`, `parse-page` deliberately stays on the fast PDF native
text baseline and no model is loaded. With `--vlm-backend paddleocr-vl`, the
pipeline loads the PaddleOCR-VL remote-code model, formats the image prompt
with the PaddleOCR-VL chat template, verifies PDF native text draft tokens, and
falls back to VLM generation after the first mismatch for blocks that are not
directly accepted. The PDF benchmark path now batches recognition across pages
after layout is prepared, so the VLM sees larger micro-batches. Use
`--verify-native-text` to force direct-accept candidates through VLM/DFlash
verification for ablations.

The PaddleOCR-VL path uses `AutoProcessor` and the model's remote-code
`PaddleOCRVLForConditionalGeneration` class through `AutoModelForCausalLM`.

For a real layout model instead of PDF text-line chunks:

```sh
uv run ocr-dflash parse-page \
  --pdf sample.pdf \
  --out-dir out/layout-vlm \
  --layout-mode pp-doclayout-v2 \
  --layout-device gpu \
  --vlm-model PaddlePaddle/PaddleOCR-VL-1.6 \
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

uv run ocr-dflash analyze-report experiment/report.json
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
`PaddleOCRVLDFlashGenerator` is the concrete PaddleOCR-VL adapter. It uses a
KV-cache chunk verification loop for greedy decoding when the model exposes
`past_key_values`, and falls back to a correctness-first recompute verifier if a
model backend cannot support cache-based verification.

PaddleOCR-VL-1.6 follows the official Paddle stack:

- `paddlepaddle-gpu==3.2.1`
- `paddleocr[doc-parser]>=3.6.0`
- `transformers>=5.0.0`

The VLM path needs a small torch overlay in the same `.venv` on this machine:

```sh
uv pip install --python .venv/bin/python --no-deps --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 nvidia-nccl-cu12==2.28.9 nvidia-nvshmem-cu12==3.4.5
```

For a quick local debug run, use `demo.py`.

## Tests

```sh
uv run pytest -q
```
