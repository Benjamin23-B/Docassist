from pathlib import Path

import pytest

from app.workers.parser import DocumentParseError, parse_file

MARKDOWN = """# Guide

Intro paragraph.

## Setup

Install it.

| Name | Value |
| --- | --- |
| a | 1 |
| b | 2 |

### Details

- point one
- point two

~~~python
print("hello")
~~~
"""


def test_parse_markdown_preserves_sections_and_tables(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(MARKDOWN, encoding="utf-8")

    blocks = parse_file(path).blocks

    texts = [block.text for block in blocks]
    assert "Intro paragraph." in texts
    assert any('print("hello")' in text for text in texts)

    intro = next(block for block in blocks if block.text == "Intro paragraph.")
    assert intro.section == "Guide"

    setup = next(block for block in blocks if block.text == "Install it.")
    assert setup.section == "Guide > Setup"

    table = next(block for block in blocks if block.kind == "table")
    assert table.section == "Guide > Setup"
    assert "| a | 1 |" in table.text

    details = next(block for block in blocks if "point one" in block.text)
    assert details.section == "Guide > Setup > Details"


def test_parse_text_splits_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")

    blocks = parse_file(path).blocks
    assert [block.text for block in blocks] == ["First paragraph.", "Second paragraph."]


def test_parse_html_extracts_headings_and_tables(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        """
        <html><body>
          <h1>Manual</h1>
          <p>Hello there.</p>
          <script>ignore_me()</script>
          <h2>Tables</h2>
          <table>
            <tr><th>A</th><th>B</th></tr>
            <tr><td>1</td><td>2</td></tr>
          </table>
        </body></html>
        """,
        encoding="utf-8",
    )

    blocks = parse_file(path).blocks
    hello = next(block for block in blocks if block.text == "Hello there.")
    assert hello.section == "Manual"

    table = next(block for block in blocks if block.kind == "table")
    assert table.section == "Manual > Tables"
    assert "A | B" in table.text
    assert all("ignore_me" not in block.text for block in blocks)


def test_parse_docx_headings_and_tables(tmp_path: Path) -> None:
    from docx import Document as DocxDocument

    path = tmp_path / "report.docx"
    document = DocxDocument()
    document.add_heading("Report title", level=1)
    document.add_paragraph("Body paragraph.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Key"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "alpha"
    table.cell(1, 1).text = "1"
    document.save(path)

    blocks = parse_file(path).blocks
    body = next(block for block in blocks if block.text == "Body paragraph.")
    assert body.section == "Report title"

    table_block = next(block for block in blocks if block.kind == "table")
    assert table_block.section == "Report title"
    assert "alpha | 1" in table_block.text


def test_parse_pdf_extracts_pages_and_headings(tmp_path: Path) -> None:
    import pymupdf

    path = tmp_path / "doc.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Large Heading", fontsize=22)
    for offset, line in enumerate(
        [
            "This is the first body line.",
            "This is the second body line.",
            "This is the third body line.",
        ]
    ):
        page.insert_text((72, 110 + offset * 16), line, fontsize=11)
    document.save(path)
    document.close()

    blocks = parse_file(path).blocks
    body = next(block for block in blocks if "first body line" in block.text)
    assert body.page_number == 1
    assert body.section == "Large Heading"


def test_parse_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK")

    with pytest.raises(DocumentParseError):
        parse_file(path)


def test_parse_rejects_corrupt_pdf(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a real pdf")

    with pytest.raises(DocumentParseError):
        parse_file(path)
