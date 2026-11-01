"""Structure-aware document parsers.

Every parser returns an ordered list of :class:`ParsedBlock` items. Each block
carries the page number and the nearest preceding heading so downstream
chunking can preserve document structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class DocumentParseError(Exception):
    """Raised when a document cannot be converted into text blocks."""


@dataclass(slots=True)
class ParsedBlock:
    text: str
    page_number: int | None = None
    section: str | None = None
    kind: str = "text"  # "text" | "table"


@dataclass(slots=True)
class ParsedDocument:
    blocks: list[ParsedBlock]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|?\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_PDF_TERMINAL_PUNCTUATION = (".", ",", ";", ":", "!")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _update_section(stack: list[tuple[int, str]], level: int, title: str) -> str:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))
    return " > ".join(text for _, text in stack)


def _current_section(stack: list[tuple[int, str]]) -> str | None:
    return " > ".join(text for _, text in stack) if stack else None


def _parse_markdown(path: Path) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    heading_stack: list[tuple[int, str]] = []
    section: str | None = None
    paragraph: list[str] = []
    table: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        text = "\n".join(paragraph).strip()
        if text:
            blocks.append(ParsedBlock(text, section=section))
        paragraph = []

    def flush_table() -> None:
        nonlocal table
        text = "\n".join(table).strip()
        if text:
            blocks.append(ParsedBlock(text, section=section, kind="table"))
        table = []

    lines = _read_text(path).splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]

        if _FENCE_RE.match(line):
            flush_paragraph()
            flush_table()
            fence = line.strip()[:3]
            code = [line]
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code.append(lines[index])
                index += 1
            if index < len(lines):
                code.append(lines[index])
            blocks.append(ParsedBlock("\n".join(code), section=section))
            index += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            flush_table()
            section = _update_section(
                heading_stack,
                len(heading.group(1)),
                heading.group(2).strip(),
            )
            index += 1
            continue

        if _TABLE_ROW_RE.match(line):
            flush_paragraph()
            table.append(line.strip())
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
            flush_table()
            index += 1
            continue

        flush_table()
        paragraph.append(line.rstrip())
        index += 1

    flush_paragraph()
    flush_table()
    return blocks


def _parse_text(path: Path) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    for piece in re.split(r"\n\s*\n", _read_text(path)):
        text = piece.strip()
        if text:
            blocks.append(ParsedBlock(text))
    return blocks


def _parse_html(path: Path) -> list[ParsedBlock]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    root = soup.body or soup
    blocks: list[ParsedBlock] = []
    heading_stack: list[tuple[int, str]] = []

    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"]):
        if element.name != "table" and element.find_parent("table") is not None:
            continue

        if element.name.startswith("h"):
            title = element.get_text(" ", strip=True)
            if title:
                _update_section(heading_stack, int(element.name[1]), title)
            continue

        if element.name == "table":
            rows = []
            for row in element.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(
                    ParsedBlock(
                        "\n".join(rows),
                        section=_current_section(heading_stack),
                        kind="table",
                    )
                )
            continue

        text = element.get_text(" ", strip=True)
        if text:
            blocks.append(
                ParsedBlock(
                    text,
                    section=_current_section(heading_stack),
                    kind="text",
                )
            )

    return blocks


def _parse_docx(path: Path) -> list[ParsedBlock]:
    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = DocxDocument(str(path))
    blocks: list[ParsedBlock] = []
    heading_stack: list[tuple[int, str]] = []

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else ""
            if style.lower().startswith(("heading", "title")):
                digits = re.findall(r"\d+", style)
                level = int(digits[0]) if digits else 1
                _update_section(heading_stack, level, text)
                continue
            blocks.append(ParsedBlock(text, section=_current_section(heading_stack)))
        elif tag == "tbl":
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(
                    ParsedBlock(
                        "\n".join(rows),
                        section=_current_section(heading_stack),
                        kind="table",
                    )
                )

    return blocks


def _join_pdf_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if text.endswith("-") and line[:1].islower():
            text = f"{text[:-1]}{line}"
        elif text:
            text = f"{text} {line}"
        else:
            text = line
    return text


def _parse_pdf(path: Path) -> list[ParsedBlock]:
    import pymupdf

    blocks: list[ParsedBlock] = []
    heading_stack: list[tuple[int, str]] = []

    with pymupdf.open(str(path)) as document:
        for page_number, page in enumerate(document, start=1):
            data = page.get_text("dict")
            page_spans = [
                span
                for block in data.get("blocks", [])
                if block.get("type") == 0
                for line in block.get("lines", [])
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            ]
            if not page_spans:
                continue

            sizes = sorted(span["size"] for span in page_spans)
            median_size = sizes[len(sizes) // 2]
            paragraph_lines: list[str] = []

            def flush_paragraph(page_number: int = page_number) -> None:
                nonlocal paragraph_lines
                text = _join_pdf_lines(paragraph_lines).strip()
                if text:
                    blocks.append(
                        ParsedBlock(
                            text,
                            page_number=page_number,
                            section=_current_section(heading_stack),
                        )
                    )
                paragraph_lines = []

            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span.get("text", "") for span in spans).strip()
                    if not text:
                        continue
                    max_size = max((span.get("size", 0) for span in spans), default=0)
                    is_heading = (
                        len(text) <= 100
                        and max_size >= median_size * 1.15
                        and not text.endswith(_PDF_TERMINAL_PUNCTUATION)
                    )
                    if is_heading:
                        flush_paragraph()
                        _update_section(heading_stack, 1, text)
                        continue
                    paragraph_lines.append(text)
                flush_paragraph()

    return blocks


_PARSERS = {
    ".md": _parse_markdown,
    ".markdown": _parse_markdown,
    ".txt": _parse_text,
    ".html": _parse_html,
    ".htm": _parse_html,
    ".docx": _parse_docx,
    ".pdf": _parse_pdf,
}


def parse_file(path: Path, extension: str | None = None) -> ParsedDocument:
    """Parse a file into structured blocks.

    Raises :class:`DocumentParseError` for unsupported or unreadable documents.
    """
    suffix = (extension or path.suffix).lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise DocumentParseError(f"Unsupported document type {suffix!r}")

    try:
        blocks = parser(path)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Failed to parse {suffix} document: {exc}") from exc

    return ParsedDocument(blocks=[block for block in blocks if block.text.strip()])
