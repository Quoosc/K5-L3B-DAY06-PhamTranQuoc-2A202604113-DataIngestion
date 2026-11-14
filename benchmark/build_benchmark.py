"""Materialize the pre-retrieval question design from verified corpus passages.

The question wording/answers below were authored by Codex from the selected
source units. This script only extracts those passages and validates references;
it does not generate questions from retrieval results or assign human labels.
Run from repository root: python benchmark/build_benchmark.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark"
PAGES_PATH = ROOT / "data/normalized/pages.jsonl"
PAGES = [json.loads(line) for line in PAGES_PATH.read_text(encoding="utf-8").splitlines()]
LOOKUP = {(p["document_id"], p["page"]): p for p in PAGES}


def passage(doc: str, page: int, start: int, end: int | None = None) -> dict:
    p = LOOKUP[doc, page]
    paragraphs = p["text"].split("\n\n")
    quote = "\n\n".join(paragraphs[start:(start + 1 if end is None else end + 1)]).strip()
    assert quote and quote in p["text"], (doc, page, start)
    return {"source_id": doc, "page": page, "evidence_quote": quote}


def containing(doc: str, page: int, needle: str) -> dict:
    paragraphs = LOOKUP[doc, page]["text"].split("\n\n")
    found = [i for i, p in enumerate(paragraphs) if needle in p]
    assert len(found) == 1, (doc, page, needle, found)
    return passage(doc, page, found[0])


QUESTIONS: list[dict] = []


def add(number: int, split: str, kind: str, question: str, answer: str,
        gold: dict, scope: str | None = None, alternatives: list[dict] | None = None) -> None:
    record = {
        "id": f"q{number:02d}", "split": split, "type": kind,
        "question": question, "expected_answer": answer,
        "gold_source_id": gold["source_id"], "gold_page": gold["page"],
        "evidence_quote": gold["evidence_quote"],
        "filter": {"document_type": scope} if scope else {},
        "alternatives": alternatives or [],
        "authorship": "Codex AI-assisted; no human review claimed",
        "human_reviewed": False,
    }
    if scope:
        record["filter_context"] = (
            f"The question explicitly requests {scope} documentation. "
            "This scope is input before retrieval, not derived from the gold source."
        )
    QUESTIONS.append(record)


add(1, "dev", "comparison",
    "How do str() and repr() differ in their intended audience when displaying a value?",
    "str() aims at a human-readable representation; repr() aims at a representation readable by the interpreter, or one that gives SyntaxError when no equivalent syntax exists.",
    containing("tutorial-inputoutput", 1, "The str function is meant"))
add(2, "dev", "filter_condition",
    "Using the tutorial guidance, why use a with statement for a file when processing might raise an exception?",
    "The file is closed when the suite finishes, including when an exception occurs.",
    containing("tutorial-inputoutput", 4, "It is good practice to use the with"), "tutorial",
    [passage("tutorial-errors", 5, 5, 6)])
add(3, "dev", "condition",
    "What seek positions are valid for a text-mode file, and what happens if another offset is used?",
    "Seek from the beginning using zero or a value returned by tell(); seek(0, 2) to the end is also allowed. Other offsets have undefined behaviour.",
    containing("tutorial-inputoutput", 5, "In text files (those opened"))
add(4, "dev", "condition",
    "Will an except clause for a derived exception class also catch instances of its base class?",
    "No. A handler matches its named class and subclasses, not instances of the named class's base classes.",
    containing("tutorial-errors", 2, "A class in an except clause matches"))
add(5, "test", "condition",
    "If both a try clause and its finally clause return a value, which return value reaches the caller?",
    "The value returned by the finally clause overrides the value from the try clause.",
    containing("tutorial-errors", 4, "* If a !finally clause includes"))
add(6, "dev", "filter_comparison",
    "In the tutorial, which collection is recommended for a queue, and why is a list inefficient at its front?",
    "Use collections.deque. Removing or inserting at the beginning of a list shifts other elements; deque supports fast appends and pops at both ends.",
    passage("tutorial-datastructures", 2, 3, 4), "tutorial")
add(7, "dev", "direct",
    "How do I create an empty set, and why would empty braces be wrong for that purpose?",
    "Use set(); {} creates an empty dictionary, not an empty set.",
    containing("tutorial-datastructures", 4, "Curly braces or the set function"))
add(8, "test", "condition",
    "Is putting a single value in parentheses sufficient to make a one-element tuple?",
    "No. Follow the value with a comma to create a one-item tuple; parentheses alone are insufficient.",
    containing("tutorial-datastructures", 4, "A special problem is the construction"))
add(9, "dev", "direct",
    "What ordering does list(d) use for dictionary keys, and how can I obtain sorted keys instead?",
    "list(d) returns keys in insertion order; use sorted(d) for sorted keys.",
    containing("tutorial-datastructures", 5, "Performing list(d)"))
add(10, "dev", "filter_condition",
    "According to the library reference, can repeated json.dump calls to the same file be used to build a valid JSON file of multiple objects?",
    "No. JSON is not a framed protocol, so repeated dump calls on the same file result in invalid JSON.",
    containing("library-json", 2, "Unlike pickle and marshal"), "library")
add(11, "dev", "parameter",
    "Which JSON encoder option lets non-ASCII characters appear directly rather than as escaped sequences?",
    "Set ensure_ascii=False. The default True escapes incoming non-ASCII characters.",
    passage("library-json", 2, 11), alternatives=[containing("library-json", 5, "If *ensure_ascii* is true")])
add(12, "test", "filter_condition",
    "Using the library reference, what happens when JSON serialization receives NaN or infinity with allow_nan=False?",
    "It raises ValueError for nan, inf and -inf, in strict compliance with the JSON specification.",
    passage("library-json", 2, 13), "library",
    [containing("library-json", 5, "If *allow_nan* is true")])
add(13, "dev", "condition",
    "When both JSON decoding callbacks object_hook and object_pairs_hook are supplied, which one takes precedence?",
    "object_pairs_hook takes priority over object_hook.",
    containing("library-json", 3, ":param object_pairs_hook:"),
    alternatives=[containing("library-json", 4, "*object_pairs_hook* is an optional function")])
add(14, "test", "direct",
    "Which JSONDecoder method can parse a JSON document followed by extra data, and what does it return?",
    "raw_decode(s) returns the decoded Python value and the index where the JSON document ended.",
    passage("library-json", 5, 2, 4))
add(15, "dev", "condition",
    "Which pathlib class can manipulate Windows paths on Unix without making filesystem calls?",
    "PureWindowsPath can be instantiated on Unix; WindowsPath cannot. Pure path classes do not access the OS.",
    passage("library-pathlib", 1, 10))
add(16, "dev", "condition",
    "When constructing a pathlib path from segments, what happens to earlier segments if a later segment is absolute?",
    "Earlier segments are ignored when an absolute segment is encountered.",
    passage("library-pathlib", 2, 1), alternatives=[passage("library-pathlib", 2, 26)])
add(17, "test", "condition",
    "On Windows, does adding a rooted relative pathlib segment such as backslash-foo reset the drive?",
    "No. The drive is not reset when a rooted relative segment is encountered.",
    passage("library-pathlib", 2, 3), alternatives=[passage("library-pathlib", 2, 26)])
add(18, "dev", "condition",
    "Why does pathlib preserve double-dot components and leading double slashes instead of simplifying them like single dots?",
    "Collapsing them could change path meaning, for example with symbolic links or UNC paths.",
    passage("library-pathlib", 2, 5))
add(19, "test", "condition",
    "What does PurePath.as_uri() do when asked to convert a relative path?",
    "It raises ValueError because the path must be absolute to be represented as a file URI.",
    passage("library-pathlib", 4, 23, 24))
add(20, "dev", "comparison",
    "What exception results from writing str to a binary stream or bytes to a text stream?",
    "Both mismatched writes raise TypeError.",
    passage("library-io", 1, 15))
add(21, "dev", "filter_direct",
    "From the library reference, how should I explicitly request UTF-8 or the current locale encoding when opening text files?",
    "Use encoding='utf-8' for UTF-8 or encoding='locale' for the current locale encoding; the latter is supported since Python 3.10.",
    passage("library-io", 2, 1), "library")
add(22, "test", "condition",
    "How does RawIOBase.read distinguish a nonblocking stream with no available bytes from end-of-file when the requested size is nonzero?",
    "No available bytes in nonblocking mode returns None; returning zero bytes with a nonzero requested size indicates EOF.",
    passage("library-io", 5, 23, 24))
add(23, "test", "direct",
    "Does IOBase.writelines insert line endings automatically between the supplied strings?",
    "No. Line separators are not added; supplied lines usually need their own separator at the end.",
    passage("library-io", 5, 14, 15))
add(24, "dev", "condition",
    "What makes an instance of a user-defined class false in a Boolean context by default?",
    "Its class must define __bool__ returning False or __len__ returning zero; otherwise objects are considered true by default.",
    passage("library-stdtypes", 1, 8))
add(25, "dev", "condition",
    "Does Python integer floor division round toward zero or negative infinity, and what is (-1)//2?",
    "It rounds toward negative infinity; (-1)//2 is -1.",
    passage("library-stdtypes", 3, 5))
add(26, "test", "condition",
    "How is int.bit_length() characterized for a nonzero integer, and what does it return for zero?",
    "For nonzero x it is the unique positive k satisfying 2**(k-1) <= abs(x) < 2**k; zero has bit_length() equal to 0.",
    passage("library-stdtypes", 4, 17))
add(27, "dev", "condition",
    "When raising a new exception with 'from None', is the old exception deleted or still available for debugging?",
    "It is hidden from the normal chained display but remains available in __context__ for introspection; __cause__ is set and __suppress_context__ becomes True.",
    passage("library-exceptions", 1, 14))
add(28, "dev", "direct",
    "What base class should application-defined exceptions normally inherit from, instead of inheriting directly from BaseException?",
    "They should normally inherit from Exception or one of its subclasses.",
    passage("library-exceptions", 1, 5),
    alternatives=[passage("tutorial-errors", 4, 4), passage("library-exceptions", 2, 19, 20)])
add(29, "test", "filter_comparison",
    "According to the library reference, how does input() reaching EOF without data differ from text-mode file read/readline reaching EOF?",
    "input() raises EOFError; the documented IOBase read/readline behavior returns an empty string at EOF.",
    passage("library-exceptions", 3, 3, 4), "library")
add(30, "dev", "condition",
    "What happens if generator code directly or indirectly raises StopIteration rather than returning normally?",
    "It is converted into RuntimeError, with the StopIteration retained as the new exception's cause.",
    passage("library-exceptions", 5, 13))


def write_json(name: str, obj: object) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    # Reproduction must preserve the original pre-retrieval timestamp. Do not
    # silently bless a modified question set or recreate a holdout after scores.
    freeze_path = OUT / "freeze.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if digest(PAGES_PATH) != frozen["corpus_sha256"]:
            raise RuntimeError("Corpus differs from the frozen benchmark; investigate before any rerun.")
        for name, expected in frozen["files"].items():
            if digest(OUT / name) != expected:
                raise RuntimeError(f"Frozen artifact changed: {name}; do not recreate an independent-test claim.")
        proposed = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in QUESTIONS)
        if proposed != (OUT / "questions.jsonl").read_text(encoding="utf-8"):
            raise RuntimeError("Question design differs from the frozen artifact; disclose any revision explicitly.")
        print("Existing benchmark freeze verified; no files or timestamps changed.")
        return
    validation = []
    for q in QUESTIONS:
        candidates = [{"source_id": q["gold_source_id"], "page": q["gold_page"],
                       "evidence_quote": q["evidence_quote"]}, *q["alternatives"]]
        refs = []
        for gold in candidates:
            p = LOOKUP[gold["source_id"], gold["page"]]
            assert gold["evidence_quote"] in p["text"]
            assert all(p["metadata"].get(k) == v for k, v in q["filter"].items())
            offset = p["text"].index(gold["evidence_quote"])
            refs.append({**gold, "source_url": p["source"], "page_offset": offset,
                         "source_offset": p["start_offset"] + offset,
                         "exact_quote_exists": True, "filter_matches": True})
        validation.append({"id": q["id"], "evidence": refs,
                           "answer_support_review": "AI-assisted reading of cited passages; human review pending",
                           "human_reviewed": False})
    assert len(QUESTIONS) == 30
    assert Counter(q["split"] for q in QUESTIONS) == {"dev": 20, "test": 10}
    assert Counter(q["split"] for q in QUESTIONS if q["filter"]) == {"dev": 4, "test": 2}
    assert len({q["question"] for q in QUESTIONS}) == 30
    (OUT / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in QUESTIONS), encoding="utf-8")

    grouped = defaultdict(list)
    for p in sorted(PAGES, key=lambda p: (p["document_id"], p["page"])):
        grouped[p["document_id"]].append(p)
    selection = []
    for index in range(max(map(len, grouped.values()))):
        for doc in sorted(grouped):
            if index < len(grouped[doc]) and len(selection) < 10:
                p = grouped[doc][index]
                spans = []
                cursor = 0
                for paragraph in p["text"].split("\n\n"):
                    if (len(paragraph.strip()) >= 80 and ">>>" not in paragraph
                            and not paragraph.strip().lower().startswith(("pair:", "single:", "triple:", "module:", "moduleauthor:", "sectionauthor:", ":", "|", "def ", "class ", "#", "with ", "for "))):
                        left = cursor + len(paragraph) - len(paragraph.lstrip())
                        spans.append((left, paragraph.strip()))
                    cursor += len(paragraph) + 2
                assert spans, (p["document_id"], p["page"])
                offset, paragraph = spans[len(spans)//2]
                selection.append({"id": f"coh{len(selection)+1:02d}", "document_id": p["document_id"],
                                  "source": p["source"], "page": p["page"], "offset": offset,
                                  "page_offset": offset, "source_offset": p["start_offset"]+offset,
                                  "anchor_text": paragraph[:160],
                                  "selection_rule": "Sorted document round-robin units; start of middle eligible prose paragraph (>=80 chars, no prompts/index fields)"})
    write_json("coherence_positions.json", selection)
    write_json("evidence_validation.json", {"question_count": 30, "split_counts": {"dev": 20, "test": 10},
               "filter_counts": {"dev": 4, "test": 2}, "all_exact_quotes_exist": True,
               "method": "Codex-authored questions; automated source/quote/filter checks; no retrieval used",
               "human_reviewed": False, "checks": validation})
    write_json("freeze.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "pre-retrieval benchmark design freeze",
               "corpus_sha256": digest(PAGES_PATH),
               "files": {name: digest(OUT/name) for name in ["questions.jsonl", "protocol.md", "coherence_positions.json", "evidence_validation.json"]},
               "retrieval_results_seen_by_author": False,
               "note": "Re-running this design builder after retrieval would not establish a new independent holdout. Preserve this initial freeze record."})
    print(f"Created 30 questions, 20 dev/10 test, 4 dev+2 test filters; {sum(1+len(q['alternatives']) for q in QUESTIONS)} verified gold passages; 10 coherence positions.")


if __name__ == "__main__":
    main()

