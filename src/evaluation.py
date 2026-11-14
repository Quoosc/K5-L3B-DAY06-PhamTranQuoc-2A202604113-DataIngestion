"""Conservative evidence retrieval metrics; never infer human audit labels."""
import re
import unicodedata
from collections import defaultdict


def normalize_evidence(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def golds(question):
    primary = {"source_id": question["gold_source_id"], "page": question["gold_page"],
               "evidence_quote": question["evidence_quote"]}
    return [primary] + question.get("alternatives", [])


def evaluate_query(question, hits):
    """A hit needs ONE chunk with a valid source, unit and full gold excerpt.

    Exact normalized evidence is a conservative proxy. Semantic equivalents need
    a separately documented review; a page match alone never counts as a hit.
    """
    evaluated = []
    for rank, hit in enumerate(hits[:3], 1):
        matches = []
        for gold in golds(question):
            source = gold.get("source_id", gold.get("gold_source_id"))
            page = gold.get("page", gold.get("gold_page"))
            quote = normalize_evidence(gold["evidence_quote"])
            if (hit["document_id"] == source and hit["page"] == page
                    and quote and quote in normalize_evidence(hit["text"])):
                matches.append({"source_id": source, "page": page, "evidence_quote": gold["evidence_quote"]})
        evaluated.append({**hit, "rank": rank, "evidence_hit": bool(matches), "matched_gold": matches})
    return {"question_id": question["id"], "split": question["split"],
            "question": question["question"], "expected_answer": question["expected_answer"],
            "requested_filter": question.get("filter", {}), "gold": golds(question),
            "hits": evaluated, "hit1": bool(evaluated and evaluated[0]["evidence_hit"]),
            "hit3": any(hit["evidence_hit"] for hit in evaluated),
            "evidence_match_method": "NFC + whitespace-normalized full excerpt in a single source/unit chunk"}


def summarize(traces):
    groups = defaultdict(list)
    for trace in traces:
        groups[(trace["strategy"], trace["split"])].append(trace)
    rows = []
    for (strategy, split), group in sorted(groups.items()):
        n = len(group)
        h1 = sum(t["hit1"] for t in group)
        h3 = sum(t["hit3"] for t in group)
        rows.append({"strategy": strategy, "split": split, "n": n,
                     "hit1_n": h1, "hit3_n": h3, "hit1": h1/n, "hit3": h3/n,
                     "returned_chunks": sum(len(t["hits"]) for t in group)})
    return rows


def filter_compliance(trace):
    requested = trace.get("requested_filter", {})
    if not requested:
        return {"applicable": False, "compliant": None, "reason": "no requested filter"}
    hits = trace["hits"]
    complies = bool(hits) and all(all(h["metadata"].get(k) == v for k, v in requested.items()) for h in hits)
    return {"applicable": True, "compliant": complies, "returned": len(hits),
            "reason": "empty retrieval is not counted as successful compliance" if not hits else "all returned hits checked"}
