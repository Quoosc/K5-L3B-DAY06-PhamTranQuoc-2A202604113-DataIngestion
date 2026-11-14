"""Reproducible ingestion of the licensed, tagged CPython RST corpus.

Source units are NOT physical PDF pages. Each selected unit is a sequential,
non-overlapping slice of its normalized document and contains >=1,500 chars.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize Unicode and line endings without destroying paragraph/code breaks."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    lines = [line.rstrip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def _inline_rst(line: str) -> str:
    # Preserve the displayed label, rather than introducing a resolved title.
    line = re.sub(r":[\w:-]+:`([^`]+)`", lambda m: re.sub(r"\s*<[^>]+>$", "", m[1]).lstrip("~"), line)
    line = re.sub(r"``([^`]+)``", r"\1", line)
    line = re.sub(r"`([^`]+)\s+<([^>]+)>`_", r"\1 (\2)", line)
    return line


def normalize_rst(raw: str) -> str:
    """Conservative text rendering; this is not a full docutils/Sphinx renderer.

    Drop link targets/index scaffolding, expose semantic directive names, and
    join hard-wrapped unindented prose. Preserve indented examples verbatim
    apart from Unicode/line endings and simple inline Sphinx roles. Remaining
    RST markup is a disclosed extraction limitation, not silently synthesized.
    """
    raw = normalize_text(raw)
    output: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if re.match(r"^\.\.\s+_[^:]+:\s*$", stripped):
            continue
        if stripped and re.fullmatch(r"[=\-~^#*+]{3,}", stripped):
            continue
        directive = re.match(r"^(\s*)\.\.\s+([\w:-]+)::\s*(.*)$", line)
        if directive:
            indent, name, value = directive.groups()
            if name in {"index", "toctree", "highlight", "default-domain"}:
                output.append("")
                continue
            if name in {"code-block", "doctest", "testcode", "testoutput", "testsetup", "testcleanup"}:
                output.append("")
                continue
            label = {"versionadded": "Version added", "versionchanged": "Version changed",
                     "deprecated": "Deprecated", "deprecated-removed": "Deprecated/removed"}.get(name, name.replace("-", " ").capitalize())
            output.extend(["", f"{indent}{label}: {_inline_rst(value)}", ""])
            continue
        if stripped.startswith(".. ") and "::" not in stripped:
            # Anonymous comment header; retain following text conservatively.
            output.append("")
            continue
        output.append(_inline_rst(line))

    joined: list[str] = []
    prose: list[str] = []

    def flush() -> None:
        if prose:
            joined.append(" ".join(prose))
            prose.clear()

    for line in output:
        if not line.strip():
            flush()
            joined.append("")
        elif line.startswith((" ", "\t")) or re.match(r"^(?:[-*+] |\d+\. |>>>|\.\.\.)", line):
            flush()
            joined.append(line)
        else:
            prose.append(line.strip())
    flush()
    return normalize_text("\n".join(joined))


def sequential_units(text: str, minimum_chars: int = 1500, target_chars: int = 3600) -> list[dict[str, Any]]:
    """Partition the entire normalized source into exact, nonoverlapping slices.

    Prefer a paragraph boundary at/after the target. Oversized paragraphs may
    extend the unit; chunking later enforces the embedding token limit. A short
    final remainder is merged into the previous source unit and never counted
    as a separate eligible page.
    """
    if minimum_chars <= 0 or target_chars < minimum_chars:
        raise ValueError("Invalid source-unit size")
    if len(text) < minimum_chars:
        return []
    boundaries = [m.end() for m in re.finditer(r"\n\n", text)]
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        if boundary - start >= target_chars:
            spans.append((start, boundary))
            start = boundary
    if start < len(text):
        if len(text) - start < minimum_chars and spans:
            spans[-1] = (spans[-1][0], len(text))
        else:
            spans.append((start, len(text)))
    return [{"page": i + 1, "start_offset": a, "end_offset": b,
             "text": text[a:b], "character_count": b - a}
            for i, (a, b) in enumerate(spans)]


def validate_pages(pages: list[dict[str, Any]], normalized_sources: dict[str, str]) -> dict[str, Any]:
    """Check the lab's source-unit rule and exact provenance, without retrieval."""
    errors: list[dict[str, Any]] = []
    seen_ids: set[tuple[str, int]] = set()
    seen_content: dict[str, tuple[str, int]] = {}
    last_end: dict[str, int] = {}
    for page in pages:
        doc, number = page["document_id"], page["page"]
        key = (doc, number)
        a, b, text = page["start_offset"], page["end_offset"], page["text"]
        if key in seen_ids:
            errors.append({"page": list(key), "error": "duplicate_page_id"})
        seen_ids.add(key)
        if len(text) < 1500:
            errors.append({"page": list(key), "error": "below_1500_characters"})
        if normalized_sources[doc][a:b] != text:
            errors.append({"page": list(key), "error": "source_span_mismatch"})
        if a < last_end.get(doc, 0):
            errors.append({"page": list(key), "error": "overlapping_source_units"})
        last_end[doc] = b
        digest = sha256_bytes(text.encode("utf-8"))
        if digest in seen_content:
            errors.append({"page": list(key), "error": "duplicate_normalized_content", "other": list(seen_content[digest])})
        seen_content[digest] = key
        for field in ("section", "document_type", "language", "version"):
            if not page["metadata"].get(field):
                errors.append({"page": list(key), "error": "missing_metadata", "field": field})
    if len(pages) < 30:
        errors.append({"error": "fewer_than_30_eligible_units", "count": len(pages)})
    return {"status": "pass" if not errors else "fail", "eligible_unit_count": len(pages),
            "minimum_character_count": min((len(p["text"]) for p in pages), default=0),
            "total_selected_characters": sum(len(p["text"]) for p in pages),
            "document_count": len({p["document_id"] for p in pages}),
            "metadata_values": {field: sorted({p["metadata"][field] for p in pages})
                                for field in ("section", "document_type", "language", "version")},
            "errors": errors}
