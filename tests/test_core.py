from pathlib import Path
import pytest

from ocr_dflash.draft_verify import (
    WhitespaceTokenizer,
    direct_accept_stats,
    stable_token_id,
    tokenize_text,
)
from ocr_dflash.eval import compare_text, levenshtein
from ocr_dflash.generator import (
    DraftVerifyingGenerator,
    GenerationOptions,
    PaddleOCRVLDFlashGenerator,
    _normalize_dflash_draft,
    _split_dflash_chunks,
)
from ocr_dflash.markdown_writer import paddle_table_tokens_to_html
from ocr_dflash.pipeline import PipelineOptions, _page_stats, run_page_pipeline
from ocr_dflash.schemas import (
    BlockRecognition,
    BoundingBox,
    DetectionResult,
    DraftVerificationStats,
    LayoutBlock,
    NativeTextCandidate,
    NativeTextQuality,
    PageDemoBlock,
    PageReport,
    PdfReport,
    PageArtifacts,
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
    torch = pytest.importorskip("torch")

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


def test_dflash_draft_normalization_matches_rust_spacing_rules():
    assert _normalize_dflash_draft("  decoder.\nThe  model ") == "decoder. The model"
    assert _normalize_dflash_draft("第 1 页") == "第1页"
    assert _normalize_dflash_draft("准确率 % 高") == "准确率%高"
    assert _normalize_dflash_draft("Vaswani∗Google") == r"Vaswani\(^{*}\)Google"
    assert _normalize_dflash_draft("memory[12]and gated") == r"memory \([12]\) and gated"
    assert _normalize_dflash_draft("Systems(NIPS 2017),Long Beach") == "Systems (NIPS 2017), Long Beach"
    assert _normalize_dflash_draft("ﬁrmly ﬂoating") == "firmly floating"


def test_dflash_chunks_isolate_special_tokens():
    class PieceTokenizer:
        pieces = {1: "alpha", 2: ",", 3: " beta", 4: ".", 5: " gamma"}

        def decode(self, ids, skip_special_tokens=False):
            _ = skip_special_tokens
            return "".join(self.pieces[int(token_id)] for token_id in ids)

    assert _split_dflash_chunks(PieceTokenizer(), [1, 2, 3, 4, 5], 2) == [[1], [2], [3], [4], [5]]


def test_cli_exposes_batch_max_pixels():
    from ocr_dflash.cli import build_parser

    parser = build_parser()
    parse_pdf = next(action for action in parser._actions if getattr(action, "dest", None) == "command").choices["parse-pdf"]
    batch_defaults = {}
    for action in parse_pdf._actions:
        if getattr(action, "dest", None) in {"batch_max_pixels", "batch_size"}:
            batch_defaults[action.dest] = action.default
    assert batch_defaults["batch_size"] == 128
    assert batch_defaults["batch_max_pixels"] == 0


def test_page_stats_report_research_metrics():
    prefix_block = PageDemoBlock(
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
    accepted_block = PageDemoBlock(
        index=2,
        class_name="text",
        label=0,
        score=1.0,
        bbox=BoundingBox(0, 0, 10, 10),
        native_text_draft="gamma",
        native_text_quality=None,
        native_text=None,
        recognition=BlockRecognition(
            backend="paddleocr-vl:dflash:accepted",
            text="gamma",
            tokens=1,
            ms=1.0,
            draft=DraftVerificationStats(
                mode="paddleocr_vl_chunk_verify",
                accepted=True,
                prefix_accepted=True,
                draft_tokens=1,
                checked_tokens=1,
                matched_tokens=1,
                accepted_tokens=1,
                rejected_tokens=0,
                rollback_tokens=0,
                generated_tokens=0,
                chunk_size=2,
            ),
        ),
        recognition_error=None,
    )
    native_block = PageDemoBlock(
        index=3,
        class_name="text",
        label=0,
        score=1.0,
        bbox=BoundingBox(0, 0, 10, 10),
        native_text_draft="delta",
        native_text_quality=None,
        native_text="delta",
        recognition=BlockRecognition(
            backend="pdf-native-text",
            text="delta",
            tokens=1,
            ms=1.0,
            draft=DraftVerificationStats(
                mode="direct_accept",
                accepted=True,
                prefix_accepted=True,
                draft_tokens=1,
                checked_tokens=1,
                matched_tokens=1,
                accepted_tokens=1,
                rejected_tokens=0,
                rollback_tokens=0,
                generated_tokens=0,
                chunk_size=2,
            ),
        ),
        recognition_error=None,
    )

    stats = _page_stats([prefix_block, accepted_block, native_block])

    assert stats["draft_coverage"] == 1.0
    assert stats["accepted_token_ratio"] == 0.75
    assert stats["rollback_ratio"] == 0.25
    assert stats["average_prefix_accepted_tokens"] == 1.0
    assert stats["prefix_accept_blocks"] == 1
    assert stats["direct_accept_blocks"] == 2
    assert stats["native_direct_accept_blocks"] == 1
    assert stats["verified_accept_blocks"] == 1


def test_page_report_separates_vlm_and_layout_timings():
    from ocr_dflash.schemas import PageArtifacts, PageReport

    report = PageReport(
        schema_version=1,
        source="sample.pdf",
        image="sample.png",
        image_size=[10, 10],
        block_count=1,
        layout_ms=2.0,
        native_text_ms=3.0,
        native_direct_ms=4.0,
        vlm_ms=5.0,
        recognition_ms=5.0,
        total_ms=14.0,
        artifacts=PageArtifacts(
            layout_json="layout.json",
            native_text_json="native.json",
            markdown="page.md",
            rendered_image="sample.png",
        ),
        blocks=[],
    )

    payload = to_jsonable(report)
    assert payload["layout_ms"] == 2.0
    assert payload["native_text_ms"] == 3.0
    assert payload["native_direct_ms"] == 4.0
    assert payload["vlm_ms"] == 5.0
    assert payload["recognition_ms"] == 5.0


def test_pdf_report_serializes_page_aggregation():
    page = PageReport(
        schema_version=1,
        source="sample.pdf",
        image="sample.png",
        image_size=[10, 10],
        block_count=1,
        layout_ms=2.0,
        native_text_ms=3.0,
        native_direct_ms=4.0,
        vlm_ms=5.0,
        recognition_ms=5.0,
        total_ms=14.0,
        artifacts=PageArtifacts(
            layout_json="layout.json",
            native_text_json="native.json",
            markdown="page.md",
            rendered_image="sample.png",
        ),
        blocks=[],
    )
    report = PdfReport(
        schema_version=1,
        source="sample.pdf",
        page_count=1,
        total_layout_ms=2.0,
        total_native_text_ms=3.0,
        total_native_direct_ms=4.0,
        total_vlm_ms=5.0,
        total_ms=14.0,
        peak_vram_mb=None,
        avg_vram_mb=None,
        pages=[page],
    )

    payload = to_jsonable(report)
    assert payload["page_count"] == 1
    assert payload["pages"][0]["vlm_ms"] == 5.0


def test_cuda_memory_stats_returns_none_without_cuda(monkeypatch):
    import ocr_dflash.pipeline as pipeline

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch())

    assert pipeline._cuda_memory_stats() == (None, None)


def test_pipeline_bypasses_vlm_for_direct_native_accept(monkeypatch, tmp_path):
    import ocr_dflash.pipeline as pipeline
    from ocr_dflash.schemas import DetectionResult, RenderedPage

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"not-an-image-needed")
    block = LayoutBlock(BoundingBox(0, 0, 10, 10), 1.0, 0, "text")

    class FailingGenerator:
        def recognize(self, image_path, block, native_candidate, options):
            raise AssertionError("VLM should be bypassed for direct native text")

    monkeypatch.setattr(
        pipeline,
        "_render_or_copy_input",
        lambda options, out_dir: RenderedPage(image_path=image_path, width=10, height=10, scale=1.0, source="test.pdf"),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_layout",
        lambda detector, image_path, image_size, layout_path: (
            DetectionResult(
                schema_version=1,
                model_id="test",
                model_revision=None,
                threshold=0.0,
                image_size=[10, 10],
                class_names=["text"],
                blocks=[block],
            ),
            0.0,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_native_text_candidates_for_blocks",
        lambda *args, **kwargs: [_candidate("alpha beta")],
    )

    report = run_page_pipeline(
        PipelineOptions(
            pdf=Path("test.pdf"),
            out_dir=tmp_path,
            generator=FailingGenerator(),
            verify_native_text=False,
        )
    )

    recognition = report.blocks[0].recognition
    assert recognition is not None
    assert recognition.backend == "pdf-native-text"
    assert recognition.text == "alpha beta"

def test_paddleocr_vl_generator_reuses_image_for_direct_dflash(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (20, 20), "white").save(image_path)

    processor = FakePaddleProcessor({"alpha": 3, "beta": 4})
    model = FakePaddleModel([3, 4], generated_suffix=[])
    generator = PaddleOCRVLDFlashGenerator(model, processor)
    requests = [
        (image_path, LayoutBlock(BoundingBox(0, 0, 10, 10), 1.0, 0, "text"), _candidate("alpha beta")),
        (image_path, LayoutBlock(BoundingBox(10, 10, 20, 20), 1.0, 0, "text"), _candidate("alpha beta")),
    ]

    results = generator.recognize_many_with_paths(requests, GenerationOptions(chunk_size=2, batch_size=2))

    assert len(results) == 2
    assert all(result is not None for result in results)
    assert [result.backend for result in results if result is not None] == [
        "paddleocr-vl:dflash:accepted",
        "paddleocr-vl:dflash:accepted",
    ]
    assert model.calls == 1


def test_pipeline_batches_pending_vlm_requests_without_class_grouping(monkeypatch, tmp_path):
    import ocr_dflash.pipeline as pipeline
    from ocr_dflash.schemas import DetectionResult, RenderedPage

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"not-an-image-needed")
    blocks = [
        LayoutBlock(BoundingBox(0, 0, 10, 10), 1.0, 0, "chart"),
        LayoutBlock(BoundingBox(0, 0, 12, 12), 1.0, 0, "text"),
    ]

    class RecordingGenerator:
        def __init__(self):
            self.calls = []

        def recognize_many(self, image_path, requests, options):
            self.calls.append((image_path, requests, options.batch_size))
            return [
                BlockRecognition(backend="paddleocr-vl:generate", text=f"t{index}", tokens=1, ms=1.0)
                for index, _ in enumerate(requests)
            ]

    generator = RecordingGenerator()

    monkeypatch.setattr(
        pipeline,
        "_render_or_copy_input",
        lambda options, out_dir: RenderedPage(image_path=image_path, width=10, height=10, scale=1.0, source="test.pdf"),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_layout",
        lambda detector, image_path, image_size, layout_path: (
            DetectionResult(
                schema_version=1,
                model_id="test",
                model_revision=None,
                threshold=0.0,
                image_size=[10, 10],
                class_names=["chart", "text"],
                blocks=blocks,
            ),
            0.0,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_native_text_candidates_for_blocks",
        lambda *args, **kwargs: [None, None],
    )

    report = pipeline.run_page_pipeline(
        PipelineOptions(
            pdf=Path("test.pdf"),
            out_dir=tmp_path,
            generator=generator,
            draft_mode="none",
        )
    )

    assert len(generator.calls) == 1
    assert len(generator.calls[0][1]) == 2
    assert [block.recognition.text for block in report.blocks if block.recognition is not None] == ["t0", "t1"]


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
        GenerationOptions(chunk_size=2),
    )

    assert recognition is not None
    assert recognition.backend == "paddleocr-vl:dflash:accepted"
    assert recognition.text == "alpha beta"
    assert recognition.draft is not None
    assert recognition.draft.accepted_tokens == 2
    assert recognition.draft.mode == "paddleocr_vl_chunk_verify"
    assert processor.last_messages[0]["content"][1]["text"] == "OCR:"
    assert processor.decode_calls == 0
    assert model.calls == 2


def test_paddleocr_vl_generator_prefix_then_generates(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    processor = FakePaddleProcessor({"alpha": 3, "beta": 4, "fixed": 5})
    model = FakePaddleModel([3, 5, 15], generated_suffix=[])
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


def test_paddleocr_vl_generator_recognize_many_batches_non_dflash(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    processor = FakePaddleProcessor({"alpha": 3, "beta": 4})
    model = FakePaddleModel([3, 4], generated_suffix=[5, 6])
    generator = PaddleOCRVLDFlashGenerator(model, processor)
    blocks = [
        LayoutBlock(BoundingBox(0, 0, 5, 5), 1.0, 0, "chart"),
        LayoutBlock(BoundingBox(5, 5, 10, 10), 1.0, 0, "image"),
    ]

    results = generator.recognize_many(
        image_path,
        [(blocks[0], None), (blocks[1], None)],
        GenerationOptions(chunk_size=2),
    )

    assert len(results) == 2
    assert all(result is not None for result in results)
    assert processor.decode_calls >= 1


def test_paddleocr_vl_generator_recognize_many_respects_batch_size(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    processor = FakePaddleProcessor({"alpha": 3, "beta": 4})
    model = FakePaddleModel([3, 4], generated_suffix=[5, 6])
    generator = PaddleOCRVLDFlashGenerator(model, processor)
    blocks = [
        LayoutBlock(BoundingBox(0, 0, 5, 5), 1.0, 0, "chart"),
        LayoutBlock(BoundingBox(5, 5, 10, 10), 1.0, 0, "chart"),
        LayoutBlock(BoundingBox(10, 10, 15, 15), 1.0, 0, "chart"),
    ]

    results = generator.recognize_many(
        image_path,
        [(block, None) for block in blocks],
        GenerationOptions(chunk_size=2, batch_size=2),
    )

    assert len(results) == 3
    assert all(result is not None for result in results)


def test_paddleocr_vl_generator_recognize_many_with_paths_batches_cross_page(tmp_path):
    from PIL import Image

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (20, 20), "white").save(image_a)
    Image.new("RGB", (20, 20), "white").save(image_b)
    processor = FakePaddleProcessor({"alpha": 3, "beta": 4})
    model = FakePaddleModel([3, 4], generated_suffix=[5, 6])
    generator = PaddleOCRVLDFlashGenerator(model, processor)
    blocks = [
        LayoutBlock(BoundingBox(0, 0, 5, 5), 1.0, 0, "chart"),
        LayoutBlock(BoundingBox(5, 5, 10, 10), 1.0, 0, "chart"),
    ]

    results = generator.recognize_many_with_paths(
        [(image_a, blocks[0], None), (image_b, blocks[1], None)],
        GenerationOptions(chunk_size=2, batch_size=2),
    )

    assert len(results) == 2
    assert all(result is not None for result in results)


def test_paddleocr_vl_generator_recognize_many_with_paths_prefetch_keeps_order(tmp_path):
    from PIL import Image

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (24, 24), "white").save(image_a)
    Image.new("RGB", (24, 24), "white").save(image_b)
    class TrackingProcessor(FakePaddleProcessor):
        def __init__(self, vocab):
            super().__init__(vocab)
            self.decode_batches = []

        def batch_decode(self, batches, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            self.decode_batches.append(len(batches))
            return [f"batch-{len(self.decode_batches)}-{index}" for index, _batch in enumerate(batches)]

    processor = TrackingProcessor({"alpha": 3, "beta": 4})
    model = FakePaddleModel([3, 4], generated_suffix=[5, 6])
    generator = PaddleOCRVLDFlashGenerator(model, processor)
    blocks = [
        LayoutBlock(BoundingBox(0, 0, 10, 10), 1.0, 0, "chart"),
        LayoutBlock(BoundingBox(1, 1, 11, 11), 1.0, 0, "chart"),
    ]

    results = generator.recognize_many_with_paths(
        [(image_a, blocks[0], None), (image_b, blocks[1], None)],
        GenerationOptions(chunk_size=2, batch_size=1),
    )

    assert len(results) == 2
    assert [result.text for result in results if result is not None] == ["batch-1-0", "batch-2-0"]
    assert processor.decode_batches == [1, 1]


def test_generation_options_exposes_batch_max_pixels():
    from ocr_dflash.generator import GenerationOptions

    assert GenerationOptions().batch_max_pixels == 0


def test_model_loader_patches_paddleocr_vl_mask_compatibility(monkeypatch):
    from ocr_dflash.model_loader import _patch_paddleocr_vl_compatibility
    torch = pytest.importorskip("torch")

    class FakeModule:
        def __init__(self):
            self.calls = []

        def create_causal_mask(self, *args, **kwargs):
            self.calls.append(kwargs)
            return kwargs

    class FakeModel:
        __module__ = "fake_module"

    fake_module = FakeModule()
    import sys

    monkeypatch.setitem(sys.modules, "fake_module", fake_module)
    _patch_paddleocr_vl_compatibility(FakeModel())
    input_embeds = torch.zeros((1, 2, 3))
    result = fake_module.create_causal_mask(config=object(), inputs_embeds=input_embeds, past_key_values=None, cache_position=1)
    assert fake_module.calls[0]["inputs_embeds"] is input_embeds
    assert "cache_position" not in fake_module.calls[0]


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
        self.decode_calls = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, return_tensors):
        torch = pytest.importorskip("torch")

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
        self.decode_calls += 1
        return self.tokenizer.batch_decode(
            batches,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )


class FakePaddleModel:
    def __init__(self, predictions, generated_suffix):
        torch = pytest.importorskip("torch")

        self.predictions = list(predictions)
        self.generated_suffix = list(generated_suffix)
        self.calls = 0
        self.weight = torch.zeros(1)
        self.config = type("Config", (), {"eos_token_id": 15})()

    def parameters(self):
        yield self.weight

    def __call__(self, **kwargs):
        torch = pytest.importorskip("torch")

        input_ids = kwargs["input_ids"]
        past = kwargs.get("past_key_values")
        use_cache = kwargs.get("use_cache", False)
        logits_to_keep = int(kwargs.get("logits_to_keep", 0) or 0)
        batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[-1])
        self.calls += 1

        start = 0 if past is None else past.pred_index
        out_len = logits_to_keep if logits_to_keep > 0 else seq_len
        logits = torch.zeros((batch_size, out_len, 16), dtype=torch.float32, device=input_ids.device)
        if past is None and logits_to_keep == 0:
            for row in range(batch_size):
                for offset in range(out_len):
                    if offset + 1 < seq_len:
                        target = int(input_ids[row, offset + 1])
                    else:
                        target = self.predictions[min(offset, len(self.predictions) - 1)]
                    logits[row, offset, target] = 1.0
        else:
            for row in range(batch_size):
                for offset in range(out_len):
                    pred_index = min(start + offset, len(self.predictions) - 1)
                    logits[row, offset, self.predictions[pred_index]] = 1.0
        cache = past or FakeCache(seq_len, pred_index=0)
        if past is not None:
            cache.length += seq_len
        cache.pred_index = start + out_len
        return type("Out", (), {"logits": logits, "past_key_values": cache if use_cache else None})

    def generate(self, **kwargs):
        torch = pytest.importorskip("torch")

        input_ids = kwargs["input_ids"]
        batch_size = int(input_ids.shape[0])
        suffix = torch.tensor([self.generated_suffix], dtype=input_ids.dtype, device=input_ids.device)
        suffix = suffix.expand(batch_size, -1)
        return torch.cat([input_ids, suffix], dim=-1)


class FakeCache:
    def __init__(self, length, pred_index=0):
        torch = pytest.importorskip("torch")

        self.length = length
        self.pred_index = pred_index
        self.tensor = torch.zeros((1, 1, max(length, 1), 1))

    def get_seq_length(self):
        return self.length

    def crop(self, max_length):
        self.length = min(self.length, max_length)

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return (self.tensor, self.tensor)
