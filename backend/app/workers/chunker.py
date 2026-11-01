"""Structure-aware chunking with token budgets and overlap.

Blocks produced by :mod:`app.workers.parser` are packed into chunks without
crossing section boundaries. Tables are kept atomic where possible, and
oversized prose/table blocks are split along sentence or row boundaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.workers.parser import ParsedBlock

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_FALLBACK_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass(slots=True)
class Chunk:
    index: int
    content: str
    page_number: int | None
    section: str | None


@dataclass(slots=True)
class _Unit:
    text: str
    page_number: int | None
    kind: str = "text"


@lru_cache(maxsize=1)
def _get_encoder() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # network/cache failure: degrade gracefully
        logger.warning(
            "tiktoken encoder unavailable (%s); falling back to approximate token counting",
            exc,
        )
        return None


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken, falling back to a word/punctuation heuristic."""
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:
            logger.debug("tiktoken encoding failed; using heuristic", exc_info=True)
    return len(_FALLBACK_TOKEN_RE.findall(text))


def chunk_blocks(
    blocks: list[ParsedBlock],
    *,
    max_tokens: int = 700,
    overlap_tokens: int = 90,
    min_tokens: int = 120,
) -> list[Chunk]:
    """Pack parsed blocks into overlapping, section-aware chunks."""
    groups = _group_by_section(blocks)
    chunks: list[Chunk] = []

    for section, section_blocks in groups:
        prefix_tokens = count_tokens(section) + 2 if section else 0
        # Reserve room for the section heading and the overlap carried forward
        # from the previous chunk so assembled chunks stay within the budget.
        effective_max = max(
            max_tokens - prefix_tokens - max(overlap_tokens, 0),
            120,
        )

        units: list[_Unit] = []
        for block in section_blocks:
            units.extend(_units_from_block(block, effective_max))

        previous_tail = ""
        for unit_group in _pack_units(units, effective_max, min_tokens):
            body = "\n\n".join(unit.text for unit in unit_group).strip()
            if not body:
                continue

            if previous_tail:
                body = f"{previous_tail}\n\n{body}"

            chunks.append(
                Chunk(
                    index=len(chunks),
                    content=_with_section(body, section),
                    page_number=unit_group[0].page_number,
                    section=section,
                )
            )

            if unit_group[-1].kind == "table":
                previous_tail = ""
            else:
                previous_tail = _tail_text(body, overlap_tokens)

    return chunks


def _group_by_section(
    blocks: list[ParsedBlock],
) -> list[tuple[str | None, list[ParsedBlock]]]:
    groups: list[tuple[str | None, list[ParsedBlock]]] = []
    for block in blocks:
        if not block.text.strip():
            continue
        if groups and groups[-1][0] == block.section:
            groups[-1][1].append(block)
        else:
            groups.append((block.section, [block]))
    return groups


def _with_section(body: str, section: str | None) -> str:
    if not section or body.lstrip().startswith(section):
        return body
    return f"{section}\n\n{body}"


def _units_from_block(block: ParsedBlock, max_tokens: int) -> list[_Unit]:
    text = block.text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [_Unit(text, block.page_number, block.kind)]
    if block.kind == "table":
        return _split_table(block, max_tokens)
    return [_Unit(piece, block.page_number, block.kind) for piece in _split_prose(text, max_tokens)]


def _split_prose(text: str, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    buffer: list[str] = []

    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if count_tokens(sentence) > max_tokens:
            if buffer:
                pieces.append(" ".join(buffer))
                buffer = []
            pieces.extend(_split_by_words(sentence, max_tokens))
            continue

        candidate = " ".join([*buffer, sentence])
        if buffer and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(buffer))
            buffer = [sentence]
        else:
            buffer.append(sentence)

    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def _split_by_words(text: str, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    buffer: list[str] = []

    for word in text.split():
        candidate = " ".join([*buffer, word])
        if buffer and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(buffer))
            buffer = [word]
        else:
            buffer.append(word)

    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def _split_table(block: ParsedBlock, max_tokens: int) -> list[_Unit]:
    rows = [row for row in block.text.splitlines() if row.strip()]
    if len(rows) <= 1:
        return [
            _Unit(piece, block.page_number, block.kind)
            for piece in _split_by_words(block.text, max_tokens)
        ]

    header, data_rows = rows[0], rows[1:]
    pieces: list[str] = []
    buffer = [header]

    for row in data_rows:
        candidate = "\n".join([*buffer, row])
        if len(buffer) > 1 and count_tokens(candidate) > max_tokens:
            pieces.append("\n".join(buffer))
            buffer = [header, row]
        else:
            buffer.append(row)

    if len(buffer) > 1 or not pieces:
        pieces.append("\n".join(buffer))

    return [_Unit(piece, block.page_number, "table") for piece in pieces]


def _pack_units(units: list[_Unit], max_tokens: int, min_tokens: int = 0) -> list[list[_Unit]]:
    if not units:
        return []

    packed: list[list[_Unit]] = []
    current: list[_Unit] = []

    for unit in units:
        if not current:
            current = [unit]
            continue

        body = "\n\n".join(item.text for item in [*current, unit])
        if count_tokens(body) <= max_tokens:
            current.append(unit)
        else:
            packed.append(current)
            current = [unit]

    if current:
        packed.append(current)

    if min_tokens > 0 and len(packed) >= 2:
        last_tokens = count_tokens("\n\n".join(u.text for u in packed[-1]))
        if last_tokens < min_tokens:
            merged = [*packed[-2], *packed[-1]]
            if count_tokens("\n\n".join(u.text for u in merged)) <= max_tokens:
                packed[-2:] = [merged]

    return packed


def _tail_text(text: str, overlap_tokens: int) -> str:
    """Return the trailing sentences of ``text`` worth up to ``overlap_tokens``."""
    if overlap_tokens <= 0:
        return ""

    sentences = [
        sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(text.strip()) if sentence.strip()
    ]
    if len(sentences) <= 1:
        sentences = _split_by_words(text, overlap_tokens) or [text.strip()]

    carry: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        size = count_tokens(sentence)
        if total + size > overlap_tokens:
            if not carry:
                words = sentence.split()
                buffer: list[str] = []
                for word in reversed(words):
                    candidate = " ".join([word, *buffer])
                    if count_tokens(candidate) > overlap_tokens:
                        break
                    buffer.insert(0, word)
                if buffer:
                    carry.insert(0, " ".join(buffer))
            break
        carry.insert(0, sentence)
        total += size

    return " ".join(carry).strip()
