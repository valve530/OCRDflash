from __future__ import annotations

from pathlib import Path

from .schemas import BlockRecognition, PageDemoBlock


def write_markdown(
    out_path: str | Path,
    source: str,
    image_path: str | Path,
    image_size: tuple[int, int],
    blocks: list[PageDemoBlock],
) -> None:
    text = render_markdown(source, image_path, image_size, blocks)
    Path(out_path).write_text(text, encoding="utf-8")


def render_markdown(
    source: str,
    image_path: str | Path,
    image_size: tuple[int, int],
    blocks: list[PageDemoBlock],
) -> str:
    width, height = image_size
    lines: list[str] = [
        "# ocr-dflash page parse",
        "",
        f"Source: `{source}`",
        "",
        f"Image: `{image_path}`",
        "",
        f"Image size: {width}x{height}",
        "",
        "## Blocks",
        "",
    ]
    for block in blocks:
        lines.extend(
            [
                (
                    f"{block.index}. **{block.class_name}** score={block.score:.4f} "
                    f"bbox=[{block.bbox.x0:.0f}, {block.bbox.y0:.0f}, "
                    f"{block.bbox.x1:.0f}, {block.bbox.y1:.0f}]"
                ),
                "",
            ]
        )
        if block.crop:
            lines.extend([f"   ![]({block.crop})", ""])
        if block.recognition is not None:
            lines.extend(_recognition_lines(block.recognition))
        elif block.native_text:
            lines.extend(["   <!-- dflash-draft: pdf-native-text accepted without VLM -->", ""])
            lines.extend(_indented_block(block.native_text))
        elif block.recognition_error:
            lines.extend([f"   <!-- recognition failed: {block.recognition_error} -->", ""])
        else:
            lines.extend([f"   <!-- {block.class_name}: no recognized text -->", ""])
        lines.append("")
    return "\n".join(lines)


def _recognition_lines(recognition: BlockRecognition) -> list[str]:
    lines = [
        f"   <!-- {recognition.backend}: {recognition.ms:.1f}ms, {recognition.tokens} tokens -->",
        "",
    ]
    lines.extend(_indented_block(render_recognition_markdown(recognition)))
    return lines


def render_recognition_markdown(recognition: BlockRecognition) -> str:
    if recognition.backend.endswith(":table"):
        table = paddle_table_tokens_to_html(recognition.text)
        if table:
            return table
    return recognition.text


def _indented_block(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return ["   <!-- empty output -->", ""]
    return [f"   {line}" for line in stripped.splitlines()] + [""]


def paddle_table_tokens_to_html(text: str) -> str | None:
    rows = parse_paddle_table_tokens(text)
    if not rows:
        return None
    rendered = ["<table>"]
    for row in rows:
        if not row:
            continue
        cells = []
        for cell_text, colspan in row:
            attr = f' colspan="{colspan}"' if colspan > 1 else ""
            cells.append(f"<td{attr}>{_escape_html(cell_text.strip())}</td>")
        rendered.append("  <tr>" + "".join(cells) + "</tr>")
    rendered.append("</table>")
    return "\n".join(rendered)


def parse_paddle_table_tokens(text: str) -> list[list[tuple[str, int]]]:
    rows: list[list[tuple[str, int]]] = []
    row: list[tuple[str, int]] = []
    cell: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "<":
            end = text.find(">", i)
            if end != -1:
                token = text[i : end + 1]
                if token in {"<ecel>", "<fcel>", "<cell>"}:
                    if cell or row:
                        row.append(("".join(cell), 1))
                        cell = []
                elif token == "<lcel>":
                    if cell or not row:
                        row.append(("".join(cell), 1))
                        cell = []
                    if row:
                        text_, colspan = row[-1]
                        row[-1] = (text_, colspan + 1)
                elif token in {"<nl>", "<tr>", "</tr>"}:
                    if cell:
                        row.append(("".join(cell), 1))
                        cell = []
                    if row:
                        rows.append(row)
                        row = []
                i = end + 1
                continue
        cell.append(text[i])
        i += 1
    if cell:
        row.append(("".join(cell), 1))
    if row:
        rows.append(row)
    return rows


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
