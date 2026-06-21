# ocr-dflash

Simple research runner for:

1. render every PDF page
2. run PP-DocLayoutV3 on CUDA
3. extract PDF native text for each layout block
4. run PaddleOCR-VL baseline `generate()` or `dflash_generate()`

The project still uses `uv` to create/run the environment, but package
installation is intentionally done with `uv pip`.

## Install

```sh
uv venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python pymupdf pillow paddleocr paddlepaddle-gpu transformers torch torchvision flash-attn
```

Use the proxy only when needed:

```sh
HTTP_PROXY=http://100.64.0.250:7890 \
HTTPS_PROXY=http://100.64.0.250:7890 \
ALL_PROXY=http://100.64.0.250:7890 \
uv pip install --python .venv/bin/python accelerate
```

Models are expected locally:

- `models/PP-DocLayoutV3`
- `models/PaddleOCR-VL-1.6`

## Run

DFlash + PDF native text:

```sh
uv run --no-sync ocr-dflash --pdf tmp/attention_1.pdf --out-dir tmp/dflash
```

If the console script has not been installed yet, use the module form:

```sh
uv run --no-sync python -m ocr_dflash.cli --pdf tmp/attention_1.pdf --out-dir tmp/dflash
```

Baseline VLM `model.generate()`:

```sh
uv run --no-sync ocr-dflash --pdf tmp/attention_1.pdf --out-dir tmp/baseline --mode baseline
```

Small debug helper:

```sh
uv run --no-sync python demo.py
```

## Important Files

- `ocr_dflash/simple_pipeline.py`: PDF -> layout -> native text -> VLM/DFlash
- `ocr_dflash/simple_vlm.py`: official baseline `generate()` and `dflash_generate()`
- `ocr_dflash/layout_demo.py`: official-style PP-DocLayoutV3 smoke
- `ocr_dflash/transformers_demo.py`: official-style PaddleOCR-VL smoke

The old broader pipeline files are left in the tree for reference while the
active CLI points at the simplified path.
