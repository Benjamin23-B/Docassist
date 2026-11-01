from app.workers.chunker import chunk_blocks, count_tokens
from app.workers.parser import ParsedBlock


def test_short_document_becomes_one_chunk() -> None:
    blocks = [ParsedBlock("Hello world. " * 10)]
    chunks = chunk_blocks(blocks, max_tokens=700, overlap_tokens=90, min_tokens=120)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].content.startswith("Hello world.")


def test_sections_do_not_mix() -> None:
    blocks = [
        ParsedBlock("Alpha content.", section="One"),
        ParsedBlock("Beta content.", section="Two"),
    ]
    chunks = chunk_blocks(blocks, max_tokens=700, overlap_tokens=90, min_tokens=120)

    assert [chunk.section for chunk in chunks] == ["One", "Two"]
    assert chunks[0].content.startswith("One")
    assert chunks[1].content.startswith("Two")


def test_long_text_is_split_within_budget_and_overlaps() -> None:
    sentences = [
        f"Sentence number {index} discusses a distinct retrieval topic." for index in range(200)
    ]
    blocks = [ParsedBlock(" ".join(sentences))]

    chunks = chunk_blocks(blocks, max_tokens=200, overlap_tokens=50, min_tokens=50)

    assert len(chunks) > 3
    for chunk in chunks:
        assert count_tokens(chunk.content) <= 200

    # The trailing sentence of one chunk is carried into the next.
    tail = chunks[0].content.split(". ")[-1]
    assert tail.strip(". ") in chunks[1].content


def test_markdown_table_is_kept_atomic() -> None:
    table = "| Name | Value |\n| --- | --- |\n| a | 1 |\n| b | 2 |"
    chunks = chunk_blocks(
        [ParsedBlock(table, kind="table")],
        max_tokens=700,
        overlap_tokens=90,
        min_tokens=120,
    )

    assert len(chunks) == 1
    assert "| --- | --- |" in chunks[0].content
    assert "| b | 2 |" in chunks[0].content


def test_oversized_table_splits_by_rows_and_repeats_header() -> None:
    rows = "\n".join(f"| {index} | value {index} |" for index in range(200))
    table = f"| id | value |\n| --- | --- |\n{rows}"

    chunks = chunk_blocks(
        [ParsedBlock(table, kind="table")],
        max_tokens=100,
        overlap_tokens=0,
        min_tokens=0,
    )

    assert len(chunks) > 1
    assert all(chunk.content.startswith("| id | value |") for chunk in chunks)
    assert all(count_tokens(chunk.content) <= 120 for chunk in chunks)


def test_page_numbers_propagate() -> None:
    blocks = [
        ParsedBlock("Page one text.", page_number=1),
        ParsedBlock("Page two text.", page_number=2),
    ]
    chunks = chunk_blocks(blocks, max_tokens=700, overlap_tokens=90, min_tokens=120)

    assert chunks[0].page_number == 1


def test_empty_blocks_produce_no_chunks() -> None:
    assert chunk_blocks([ParsedBlock("   ")], max_tokens=700) == []
    assert chunk_blocks([], max_tokens=700) == []


def test_count_tokens_handles_empty_text() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0
