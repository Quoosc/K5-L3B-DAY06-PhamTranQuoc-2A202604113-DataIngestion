"""Token-aware chunking with exact offsets into each normalized source unit.

A changes both boundaries and overlap compared with B. C deliberately uses B's
identical chunks; only the retrieval filter differs. No gold data is consulted.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any


def token_count(text: str, tokenizer: Any) -> int:
    """Count content tokens without special tokens or an embedding prefix."""
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _fixed_spans(text: str, tokenizer: Any, max_tokens: int,
                 overlap: int) -> list[tuple[int, int]]:
    """Retokenize each remaining slice to guard against boundary token changes.

    Offset mappings are character offsets, not decoded-token approximations.
    This preserves punctuation and Unicode exactly in the original source.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        tail = text[cursor:]
        encoded = tokenizer(tail, add_special_tokens=False,
                            return_offsets_mapping=True, truncation=False)
        offsets = [(int(a), int(b)) for a, b in encoded['offset_mapping'] if b > a]
        if not offsets:
            if tail.strip():
                raise ValueError('Nonempty text has no usable tokenizer offsets')
            break
        take = min(max_tokens, len(offsets))
        end = len(tail) if take == len(offsets) else offsets[take][0]
        # A substring can tokenize differently at its last boundary. Never rely
        # on the full-page token count to silently truncate the stored text.
        while take > 0 and (end <= 0 or token_count(tail[:end], tokenizer) > max_tokens):
            take -= 1
            end = offsets[take][0] if take else 0
        if end <= 0:
            raise ValueError('Cannot form a nonempty chunk within the token budget')
        start_abs, end_abs = _trim_span(text, cursor, cursor + end)
        if start_abs < end_abs:
            spans.append((start_abs, end_abs))
        if take == len(offsets):
            break
        next_local = offsets[max(0, take - overlap)][0] if overlap else end
        # A pathological Unicode offset may be shared by multiple tokens.
        if next_local <= 0:
            next_local = end
        if next_local <= 0:
            raise ValueError('Chunking did not advance')
        cursor += next_local
    return spans


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    result = []
    cursor = start
    for boundary in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9"`])', text[start:end]):
        boundary_start, boundary_end = start + boundary.start(), start + boundary.end()
        a, b = _trim_span(text, cursor, boundary_start)
        if a < b:
            result.append((a, b))
        cursor = boundary_end
    a, b = _trim_span(text, cursor, end)
    if a < b:
        result.append((a, b))
    return result


def _paragraph_spans(text: str, tokenizer: Any, max_tokens: int) -> list[tuple[int, int]]:
    paragraphs: list[tuple[int, int]] = []
    cursor = 0
    for boundary in re.finditer(r'\n[ \t]*\n+', text):
        a, b = _trim_span(text, cursor, boundary.start())
        if a < b:
            paragraphs.append((a, b))
        cursor = boundary.end()
    a, b = _trim_span(text, cursor, len(text))
    if a < b:
        paragraphs.append((a, b))

    units: list[tuple[int, int]] = []
    for a, b in paragraphs:
        if token_count(text[a:b], tokenizer) <= max_tokens:
            units.append((a, b))
            continue
        for c, d in _sentence_spans(text, a, b):
            if token_count(text[c:d], tokenizer) <= max_tokens:
                units.append((c, d))
            else:
                units.extend((c + x, c + y) for x, y in
                             _fixed_spans(text[c:d], tokenizer, max_tokens, 0))

    grouped: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for a, b in units:
        if current is None:
            current = (a, b)
        elif token_count(text[current[0]:b], tokenizer) <= max_tokens:
            current = (current[0], b)
        else:
            grouped.append(current)
            current = (a, b)
    if current is not None:
        grouped.append(current)
    return grouped


def chunk_pages(pages: list[dict], strategy: str, tokenizer: Any,
                max_tokens: int = 320, overlap: int = 48) -> list[dict]:
    """Build chunks from page dictionaries, keeping page-relative char offsets.

    B and C use paragraph-aware splitting with zero overlap regardless of the
    fixed-size overlap setting. Calling C produces the same IDs and text as B.
    """
    if strategy not in {'A', 'B', 'C'}:
        raise ValueError('strategy must be A, B or C')
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 2:
        raise ValueError('max_tokens must be an integer >= 2')
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < max_tokens:
        raise ValueError('overlap must satisfy 0 <= overlap < max_tokens')
    policy = 'A' if strategy == 'A' else 'B'
    effective_overlap = overlap if policy == 'A' else 0
    chunks = []
    page_ids: set[tuple[str, int]] = set()
    for page in pages:
        for key in ('document_id', 'source', 'text'):
            if not isinstance(page.get(key), str):
                raise ValueError(f'page.{key} must be a string')
        if not page['document_id'] or not page['source']:
            raise ValueError('document_id and source must not be empty')
        if isinstance(page.get('page'), bool) or not isinstance(page.get('page'), int) or page['page'] < 1:
            raise ValueError('page.page must be a positive integer')
        key = (page['document_id'], page['page'])
        if key in page_ids:
            raise ValueError(f'Duplicate normalized source unit: {key}')
        page_ids.add(key)
        if not isinstance(page.get('metadata', {}), dict):
            raise ValueError('page.metadata must be an object')
        text = page['text']
        if not text.strip():
            continue  # ingestion owns the explicit empty-page error log
        spans = (_fixed_spans(text, tokenizer, max_tokens, effective_overlap)
                 if policy == 'A' else _paragraph_spans(text, tokenizer, max_tokens))
        for start, end in spans:
            content = text[start:end]
            count = token_count(content, tokenizer)
            if count > max_tokens or not content.strip():
                raise AssertionError('Invalid emitted chunk')
            identity = json.dumps({
                'policy': policy, 'max_tokens': max_tokens, 'overlap': effective_overlap,
                'document_id': page['document_id'], 'source': page['source'],
                'page': page['page'], 'start': start, 'end': end, 'text': content,
            }, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            chunks.append({
                'chunk_id': hashlib.sha256(identity.encode('utf-8')).hexdigest(),
                'document_id': page['document_id'], 'source': page['source'],
                'page': page['page'], 'text': content,
                'metadata': deepcopy(page.get('metadata', {})),
                'start_offset': start, 'end_offset': end,
                'token_count': count, 'chunk_policy': policy,
            })
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise AssertionError('Duplicate chunk identity')
    return chunks
