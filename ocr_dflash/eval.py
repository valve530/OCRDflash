from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_utils import read_json, write_json


@dataclass(slots=True)
class TextMetrics:
    char_accuracy: float
    edit_distance: int
    expected_chars: int
    actual_chars: int
    exact_match: bool


def compare_text(expected: str, actual: str) -> TextMetrics:
    distance = levenshtein(expected, actual)
    denom = max(len(expected), 1)
    return TextMetrics(
        char_accuracy=max(0.0, 1.0 - distance / denom),
        edit_distance=distance,
        expected_chars=len(expected),
        actual_chars=len(actual),
        exact_match=expected == actual,
    )


def compare_text_files(expected_path: str | Path, actual_path: str | Path) -> TextMetrics:
    expected = Path(expected_path).read_text(encoding="utf-8")
    actual = Path(actual_path).read_text(encoding="utf-8")
    return compare_text(expected, actual)


def compare_report_blocks(expected_report: str | Path, actual_report: str | Path) -> dict[str, object]:
    expected = read_json(expected_report)
    actual = read_json(actual_report)
    expected_blocks = expected.get("blocks", [])
    actual_blocks = actual.get("blocks", [])
    matches = 0
    text_metrics: list[dict[str, object]] = []
    for expected_block, actual_block in zip(expected_blocks, actual_blocks):
        expected_text = _block_text(expected_block)
        actual_text = _block_text(actual_block)
        metric = compare_text(expected_text, actual_text)
        if metric.exact_match:
            matches += 1
        text_metrics.append(
            {
                "char_accuracy": metric.char_accuracy,
                "edit_distance": metric.edit_distance,
                "exact_match": metric.exact_match,
            }
        )
    block_count = max(len(expected_blocks), len(actual_blocks), 1)
    return {
        "block_count_expected": len(expected_blocks),
        "block_count_actual": len(actual_blocks),
        "block_exact_matches": matches,
        "block_exact_match_ratio": matches / block_count,
        "text_metrics": text_metrics,
    }


def write_text_metrics(out_path: str | Path, metrics: TextMetrics | dict[str, object]) -> None:
    write_json(out_path, metrics)


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        current = [i]
        for j, right_ch in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_ch != right_ch)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _block_text(block: dict[str, object]) -> str:
    recognition = block.get("recognition")
    if isinstance(recognition, dict):
        text = recognition.get("text")
        if isinstance(text, str):
            return text
    native_text = block.get("native_text")
    if isinstance(native_text, str):
        return native_text
    draft = block.get("native_text_draft")
    return draft if isinstance(draft, str) else ""
