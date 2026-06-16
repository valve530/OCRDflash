from pathlib import Path

from ocr_dflash.draft_verify import (
    WhitespaceTokenizer,
    direct_accept_stats,
    stable_token_id,
    tokenize_text,
)
from ocr_dflash.eval import compare_text, levenshtein
from ocr_dflash.generator import DraftVerifyingGenerator, GenerationOptions, PaddleOCRVLDFlashGenerator
from ocr_dflash.markdown_writer import paddle_table_tokens_to_html
from ocr_dflash.pipeline import _page_stats
from ocr_dflash.schemas import (
    BlockRecognition,
    BoundingBox,
    DetectionResult,
    DraftVerificationStats,
    LayoutBlock,
    NativeTextCandidate,
    NativeTextQuality,
    PageDemoBlock,
    to_jsonable,
)


def test_direct_accept_stats_counts_tokens():
    stats = direct_accept_stats("alpha beta", WhitespaceTokenizer(), chunk_size=4)

    assert stats.accepted
    assert stats.draft_tokens == 2
    assert stats.accepted_tokens == 2
    assert stats.chunk_size == 4


def test_stable_token_id_is_stable():
    assert stable_token_id("draft") == stable_token_id("draft")
    assert stable_token_id("draft") != stable_token_id("verify")


def test_table_tokens_render_as_html():
    html = paddle_table_tokens_to_html("A<ecel>B<nl>C")

    assert html is not None
    assert "<table>" in html
    assert "<td>A</td>" in html
    assert "<td>B</td>" in html


def test_text_eval_metrics():
    assert levenshtein("kitten", "sitting") == 3
    metrics = compare_text("abcd", "abxd")
    assert metrics.edit_distance == 1
    assert metrics.char_accuracy == 0.75
    assert not metrics.exact_match


def test_detection_result_serializes_to_rust_shaped_json():
    result = DetectionResult(
        schema_version=1,
        model_id="test",
        model_revision=None,
        threshold=0.5,
        image_size=[100, 80],
        class_names=["text"],
        blocks=[
            LayoutBlock(
                bbox=BoundingBox(1.0, 2.0, 3.0, 4.0),
                score=0.9,
                label=0,
                class_name="text",
            )
        ],
    )

    data = to_jsonable(result)

    assert data["blocks"][0]["bbox"]["x0"] == 1.0
    assert data["class_names"] == ["text"]


class SequenceModel:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.index = 0

    def next_token_id(self, context_ids):
        _ = context_ids
        value = self.sequence[self.index]
        self.index += 1
        return value


def test_draft_verifying_generator_accepts_full_draft():
    tokenizer = WhitespaceTokenizer()
    draft = "alpha beta"
    model = SequenceModel(tokenize_text(tokenizer, draft))
    generator = DraftVerifyingGenerator(model, tokenizer)

    recognition = generator.recognize(
        Path("/tmp/nonexistent.png"),
        LayoutBlock(BoundingBox(0, 0, 10, 10), 1.0, 0, "text"),
        _candidate(draft),
        GenerationOptions(chunk_size=2),
    )

    assert recognition is not None
    assert recognition.backend == "draft-verified:accepted"
    assert recognition.draft is not None
    assert recognition.draft.accepted_tokens == 2


def test_draft_verifying_generator_reports_prefix_reject():
    tokenizer = WhitespaceTokenizer()
    draft_ids = tokenize_text(tokenizer, "alpha beta")
    model = SequenceModel([draft_ids[0], stable_token_id("wrong")])
    generator = DraftVerifyingGenerator(model, tokenizer)

    recognition = generator.recognize(
        Path("/tmp/nonexistent.png"),
        LayoutBlock(BoundingBox(0, 0, 10, 10), 1.0, 0, "text"),
        _candidate("alpha beta"),
        GenerationOptions(chunk_size=2),
    )

    assert recognition is not None
    assert recognition.backend == "draft-verified:fallback-native"
    assert recognition.draft is not None
    assert recognition.draft.prefix_accepted
    assert recognition.draft.accepted_tokens == 1
    assert recognition.draft.rollback_tokens == 1


def test_transformers_next_token_verifier_uses_last_logits():
    import torch

    from ocr_dflash.transformers_verify import TransformersNextTokenVerifier

    class TinyModel:
        def parameters(self):
            yield torch.zeros(1)

        def __call__(self, input_ids):
            _ = input_ids
            logits = torch.zeros((1, 2, 5))
            logits[0, -1, 3] = 9.0
            return type("Out", (), {"logits": logits})

    verifier = TransformersNextTokenVerifier(TinyModel())

    assert verifier.next_token_id([1, 2]) == 3


def test_page_stats_report_research_metrics():
    block = PageDemoBlock(
        index=1,
        class_name="text",
        label=0,
        score=1.0,
        bbox=BoundingBox(0, 0, 10, 10),
        native_text_draft="alpha beta",
        native_text_quality=NativeTextQuality(
            char_count=10,
            char_area_ratio=1.0,
            native_bbox_area_ratio=1.0,
            width_coverage=1.0,
            height_coverage=1.0,
            line_count=1,
            direct_accept=False,
            direct_accept_reason="test",
        ),
        native_text="alpha beta",
        recognition=BlockRecognition(
            backend="draft-verified:prefix",
            text="alpha beta",
            tokens=2,
            ms=1.0,
            draft=DraftVerificationStats(
                mode="token_verify",
                accepted=False,
                prefix_accepted=True,
                draft_tokens=2,
                checked_tokens=2,
                matched_tokens=1,
                accepted_tokens=1,
                rejected_tokens=1,
                rollback_tokens=1,
                generated_tokens=1,
                chunk_size=2,
            ),
        ),
        recognition_error=None,
    )

    stats = _page_stats([block])

    assert stats["draft_coverage"] == 1.0
    assert stats["accepted_token_ratio"] == 0.5
    assert stats["rollback_ratio"] == 0.5
    assert stats["average_prefix_accepted_tokens"] == 1.0
    assert stats["prefix_accept_blocks"] == 1


def test_model_loader_returns_transformers_generator(monkeypatch):
    import types

    import ocr_dflash.model_loader as model_loader

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=FakeProcessor,
        AutoModelForImageTextToText=FakeModel,
        AutoModelForCausalLM=FakeModel,
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    generator = model_loader.load_transformers_vlm("fake-model", device="cpu", dtype="fp32")

    assert generator.model.__class__ is FakeModel
    assert generator.processor.__class__ is FakeProcessor


def test_paddleocr_vl_generator_accepts_full_draft(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    processor = FakePaddleProcessor({"alpha": 3, "beta": 4})
    model = FakePaddleModel([3, 4], generated_suffix=[])
    generator = PaddleOCRVLDFlashGenerator(model, processor)

    recognition = generator.recognize(
        image_path,
        LayoutBlock(BoundingBox(0, 0, 10, 10), 1.0, 0, "text"),
        _candidate("alpha beta"),
        GenerationOptions(chunk_size=1),
    )

    assert recognition is not None
    assert recognition.backend == "paddleocr-vl:dflash:accepted"
    assert recognition.text == "alpha beta"
    assert recognition.draft is not None
    assert recognition.draft.accepted_tokens == 2
    assert processor.last_messages[0]["content"][1]["text"] == "OCR:"


def test_paddleocr_vl_generator_prefix_then_generates(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    processor = FakePaddleProcessor({"alpha": 3, "beta": 4, "fixed": 5})
    model = FakePaddleModel([3, 9], generated_suffix=[5])
    generator = PaddleOCRVLDFlashGenerator(model, processor)

    recognition = generator.recognize(
        image_path,
        LayoutBlock(BoundingBox(0, 0, 10, 10), 1.0, 0, "text"),
        _candidate("alpha beta"),
        GenerationOptions(chunk_size=2, max_tokens=1),
    )

    assert recognition is not None
    assert recognition.backend == "paddleocr-vl:dflash:prefix"
    assert recognition.text == "alpha fixed"
    assert recognition.draft is not None
    assert recognition.draft.accepted_tokens == 1
    assert recognition.draft.generated_tokens == 1


def test_paddleocr_layout_item_parsing_normalizes_dicts():
    from ocr_dflash.layout import _layout_block_from_paddleocr

    block = _layout_block_from_paddleocr(
        {"bbox": [1, 2, 30, 40], "label": "Plain Text", "score": 0.9, "class_id": 7},
        index=0,
        image_width=100,
        image_height=100,
    )

    assert block is not None
    assert block.class_name == "text"
    assert block.label == 7
    assert block.score == 0.9
    assert block.bbox.x1 == 30


def test_paddleocr_layout_item_parsing_accepts_polygons():
    from ocr_dflash.layout import _layout_block_from_paddleocr

    block = _layout_block_from_paddleocr(
        {"poly": [[1, 2], [30, 2], [30, 40], [1, 40]], "category": "Table"},
        index=3,
        image_width=20,
        image_height=100,
    )

    assert block is not None
    assert block.class_name == "table"
    assert block.bbox.x1 == 20
    assert block.label == 3


def _candidate(text: str) -> NativeTextCandidate:
    return NativeTextCandidate(
        text=text,
        quality=NativeTextQuality(
            char_count=len(text),
            char_area_ratio=1.0,
            native_bbox_area_ratio=1.0,
            width_coverage=1.0,
            height_coverage=1.0,
            line_count=1,
            direct_accept=True,
            direct_accept_reason="test",
        ),
    )


class FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse = {value: key for key, value in vocab.items()}

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return [self.vocab[piece] for piece in text.split()]

    def batch_decode(self, batches, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        _ = skip_special_tokens, clean_up_tokenization_spaces
        return [" ".join(self.inverse.get(int(token), f"<{int(token)}>") for token in batch) for batch in batches]


class FakePaddleProcessor:
    def __init__(self, vocab):
        self.tokenizer = FakeTokenizer(vocab)
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, return_tensors):
        import torch

        assert tokenize and add_generation_prompt and return_dict and return_tensors == "pt"
        self.last_messages = messages
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 1), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def batch_decode(self, batches, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return self.tokenizer.batch_decode(
            batches,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )


class FakePaddleModel:
    def __init__(self, predictions, generated_suffix):
        import torch

        self.predictions = list(predictions)
        self.generated_suffix = list(generated_suffix)
        self.calls = 0
        self.weight = torch.zeros(1)

    def parameters(self):
        yield self.weight

    def __call__(self, **kwargs):
        import torch

        _ = kwargs
        predicted = self.predictions[self.calls]
        self.calls += 1
        logits = torch.zeros((1, 1, 16))
        logits[0, 0, predicted] = 1.0
        return type("Out", (), {"logits": logits})

    def generate(self, **kwargs):
        import torch

        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([self.generated_suffix], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=-1)
