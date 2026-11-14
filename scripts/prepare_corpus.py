"""Fetch/cache a licensed CPython snapshot and produce a traceable 40-unit corpus.

Run from repository root: python scripts/prepare_corpus.py
Use --offline to rebuild from checked-in raw files without network.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingestion import normalize_rst, sequential_units, sha256_bytes, validate_pages

VERSION = "3.12.10"
TAG = "v" + VERSION
BASE = f"https://raw.githubusercontent.com/python/cpython/{TAG}/"
SOURCES = [
    ("tutorial-inputoutput", "Doc/tutorial/inputoutput.rst", "Input and Output", "tutorial", "inputoutput"),
    ("tutorial-errors", "Doc/tutorial/errors.rst", "Errors and Exceptions", "tutorial", "errors"),
    ("tutorial-datastructures", "Doc/tutorial/datastructures.rst", "Data Structures", "tutorial", "datastructures"),
    ("library-json", "Doc/library/json.rst", "JSON encoder and decoder", "library", "json"),
    ("library-pathlib", "Doc/library/pathlib.rst", "Object-oriented filesystem paths", "library", "pathlib"),
    ("library-io", "Doc/library/io.rst", "Core tools for working with streams", "library", "io"),
    ("library-stdtypes", "Doc/library/stdtypes.rst", "Built-in Types", "library", "stdtypes"),
    ("library-exceptions", "Doc/library/exceptions.rst", "Built-in Exceptions", "library", "exceptions"),
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Require cached source files, never fetch")
    args = parser.parse_args()
    for directory in ("data/raw", "data/normalized/sources", "data/licenses", "results"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    access_path = ROOT / "data/raw/access_log.json"
    access = json.loads(access_path.read_text(encoding="utf-8")) if access_path.exists() else {}
    errors: list[dict] = []
    jobs = [(path, ROOT / "data/raw" / (doc + ".rst")) for doc, path, *_ in SOURCES]
    jobs += [("LICENSE", ROOT / "data/licenses/CPYTHON_LICENSE.txt"),
             ("Doc/license.rst", ROOT / "data/licenses/CPYTHON_DOCUMENTATION_LICENSE.rst")]

    def fetch(job: tuple[str, Path]) -> tuple[str, dict]:
        relative, target = job
        url = BASE + relative
        if target.exists():
            data = target.read_bytes()
            record = access.get(relative)
            if not record:
                raise RuntimeError(f"Cached file lacks provenance record: {target}")
            if sha256_bytes(data) != record["sha256"]:
                raise RuntimeError(f"Cached source checksum mismatch: {target}")
            return relative, record
        if args.offline:
            raise FileNotFoundError(f"Offline source missing: {target}")
        for attempt in range(1, 4):
            try:
                with urlopen(Request(url, headers={"User-Agent": "Lab06-CorpusPreparation/1.0"}), timeout=45) as response:
                    data = response.read()
                data.decode("utf-8")
                target.write_bytes(data)
                return relative, {"url": url, "path": target.relative_to(ROOT).as_posix(),
                                  "accessed_at": datetime.now(timezone.utc).isoformat(),
                                  "sha256": sha256_bytes(data), "bytes": len(data), "snapshot": TAG}
            except Exception as exc:
                errors.append({"stage": "download", "source": url, "attempt": attempt,
                               "error_type": type(exc).__name__, "message": str(exc)})
                if attempt == 3:
                    raise
                time.sleep(attempt)
        raise AssertionError("unreachable")

    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for relative, record in pool.map(fetch, jobs):
                access[relative] = record
        write_json(access_path, access)
        pages: list[dict] = []
        manifests: list[dict] = []
        normalized_sources: dict[str, str] = {}
        for document_id, relative, title, kind, section in SOURCES:
            raw_path = ROOT / "data/raw" / (document_id + ".rst")
            raw = raw_path.read_text(encoding="utf-8")
            normalized = normalize_rst(raw)
            normalized_path = ROOT / "data/normalized/sources" / (document_id + ".txt")
            normalized_path.write_text(normalized, encoding="utf-8", newline="\n")
            normalized_sources[document_id] = normalized
            units = sequential_units(normalized)
            if len(units) < 5:
                raise ValueError(f"Source has fewer than five eligible units: {document_id} ({len(units)})")
            selected = units[:5]
            metadata = {"section": section, "document_type": kind, "language": "en", "version": VERSION}
            for unit in selected:
                pages.append({"document_id": document_id, "source": BASE + relative,
                              "page": unit["page"], "text": unit["text"], "metadata": metadata,
                              "start_offset": unit["start_offset"], "end_offset": unit["end_offset"],
                              "character_count": unit["character_count"]})
            manifests.append({"source_id": document_id, "document_id": document_id, "title": title,
                              "source_path_or_url": BASE + relative, "snapshot": TAG,
                              "license_or_permission": "Python Software Foundation License Version 2; retain upstream notices; code examples additionally have documentation license terms",
                              "license_files": ["data/licenses/CPYTHON_LICENSE.txt", "data/licenses/CPYTHON_DOCUMENTATION_LICENSE.rst"],
                              "accessed_at": access[relative]["accessed_at"], "raw_path": raw_path.relative_to(ROOT).as_posix(),
                              "raw_sha256": access[relative]["sha256"], "normalized_path": normalized_path.relative_to(ROOT).as_posix(),
                              "normalized_sha256": sha256_bytes(normalized.encode("utf-8")),
                              "normalized_character_count": len(normalized), "metadata": metadata,
                              "eligible_unit_count_before_selection": len(units), "selected_unit_count": len(selected),
                              "units": [{k: v for k, v in unit.items() if k != "text"} | {
                                  "selected": unit["page"] <= 5,
                                  "selection_reason": "first_five_sequential_units_per_source" if unit["page"] <= 5 else "outside_predeclared_corpus_scope_not_counted"}
                                  for unit in units]})
        pages_path = ROOT / "data/normalized/pages.jsonl"
        pages_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages), encoding="utf-8", newline="\n")
        validation = validate_pages(pages, normalized_sources)
        validation["unit_definition"] = "sequential normalized RST text slices, >=1500 characters, exact [start_offset,end_offset); not physical PDF pages"
        validation["selection_rule"] = "first five sequential eligible units from each of eight fixed documents, chosen before retrieval/benchmark"
        validation["pages_sha256"] = sha256_bytes(pages_path.read_bytes())
        validation["blank_source_count"] = 0
        validation["duplicate_content_count"] = sum(e["error"] == "duplicate_normalized_content" for e in validation["errors"])
        write_json(ROOT / "data/manifest.json", {"corpus_id": "cpython-3.12.10-docs-first-five-units-v1",
                   "source_repository": "https://github.com/python/cpython", "snapshot": TAG,
                   "counting_rule": validation["unit_definition"], "selection_rule": validation["selection_rule"],
                   "normalization": "NFC; LF; normalized whitespace preserving paragraphs/code; conservative RST role/directive cleanup; no OCR",
                   "source_unit_target_chars": 3600, "source_unit_min_chars": 1500,
                   "selected_unit_count": len(pages), "total_selected_characters": validation["total_selected_characters"],
                   "normalized_pages_path": "data/normalized/pages.jsonl", "pages_sha256": validation["pages_sha256"],
                   "sources": manifests})
        write_json(ROOT / "results/corpus_validation.json", validation)
        if validation["status"] != "pass":
            raise ValueError(f"Corpus validation failed: {validation['errors']}")
        print(json.dumps(validation, ensure_ascii=False, indent=2))
    except Exception as exc:
        errors.append({"stage": "prepare_corpus", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        (ROOT / "results/ingestion_errors.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errors), encoding="utf-8")


if __name__ == "__main__":
    main()
