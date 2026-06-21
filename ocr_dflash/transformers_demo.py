from PIL import Image
import torch

try:
    from transformers import masking_utils

    _original_create_causal_mask = masking_utils.create_causal_mask

    def _patched_create_causal_mask(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return _original_create_causal_mask(*args, **kwargs)

    masking_utils.create_causal_mask = _patched_create_causal_mask
    print("Applied create_causal_mask compatibility patch")

except Exception as e:
    print(f"Warning: Could not apply create_causal_mask patch: {e}")
from transformers import AutoProcessor, AutoModelForCausalLM

# ---- Settings ----
model_path = "models/PaddleOCR-VL-1.6"
image_path = "/home/valve/ocr_dflash/tmp/attention_1_layout_debug/page_0000.png"
task = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula' | 'spotting' | 'seal'
# ------------------

# ---- Image Preprocessing For Spotting ----
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
spotting_upscale_threshold = 1500

if (
    task == "spotting"
    and orig_w < spotting_upscale_threshold
    and orig_h < spotting_upscale_threshold
):
    process_w, process_h = orig_w * 2, orig_h * 2
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    image = image.resize((process_w, process_h), resample_filter)

# Set max_pixels: use 1605632 for spotting, otherwise use default ~1M pixels
max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
# ---------------------------

# -------- Inference --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATTN_IMPLEMENTATION = "flash_attention_2" if DEVICE == "cuda" else None
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

model = (
    AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        # attn_implementation=ATTN_IMPLEMENTATION,
        trust_remote_code=True,
    )
    .to(DEVICE)
    .eval()
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[task]},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    images_kwargs={
        "size": {
            "shortest_edge": processor.image_processor.min_pixels,
            "longest_edge": max_pixels,
        }
    },
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
result = processor.decode(
    outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
)
print(result)
# ---------------------------
