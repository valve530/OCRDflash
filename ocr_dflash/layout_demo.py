from pathlib import Path

from paddleocr import LayoutDetection


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "PP-DocLayoutV3"
IMAGE_PATH = ROOT / "tmp" / "attention_1_layout_debug" / "page_0000.png"
OUTPUT_DIR = ROOT / "tmp" / "layout" / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = LayoutDetection(model_name="PP-DocLayoutV3", model_dir=str(MODEL_DIR), device="gpu")
    output = model.predict(
        str(IMAGE_PATH),
        batch_size=1,
        layout_nms=True,
    )
    for res in output:
        res.print()
        res.save_to_img(save_path=str(OUTPUT_DIR))
        res.save_to_json(save_path=str(OUTPUT_DIR / "res.json"))


if __name__ == "__main__":
    main()
