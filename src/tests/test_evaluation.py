from src.evaluation import evaluate_query, summarize, filter_compliance


def question():
    return {"id":"q1", "split":"dev", "question":"When?", "expected_answer":"Only if X.",
            "gold_source_id":"doc", "gold_page":1, "evidence_quote":"Allowed only if X.", "filter":{}}


def hit(text="Allowed only if X.", source="doc", page=1):
    return {"chunk_id":"c1", "document_id":source, "page":page, "text":text, "metadata":{}}


def test_full_evidence_in_correct_source_required():
    q=question()
    assert evaluate_query(q,[hit()])["hit1"]
    assert not evaluate_query(q,[hit(source="other")])["hit3"]
    assert not evaluate_query(q,[hit(page=2)])["hit3"]
    assert not evaluate_query(q,[hit("Allowed")])["hit3"]


def test_split_evidence_does_not_count_union():
    assert not evaluate_query(question(),[hit("Allowed only"),hit("if X.")])["hit3"]


def test_alt_gold_and_top3_only():
    q=question(); q["alternatives"]=[{"source_id":"other","page":2,"evidence_quote":"Alternative evidence."}]
    assert evaluate_query(q,[hit("Alternative evidence.","other",2)])["hit3"]
    assert not evaluate_query(q,[hit("no")]*3+[hit()])["hit3"]


def test_empty_and_short_results_keep_denominator():
    a=evaluate_query(question(),[]); b=evaluate_query(question(),[hit()])
    a["strategy"]=b["strategy"]="A"
    row=summarize([a,b])[0]
    assert row["n"]==2 and row["hit3"]==0.5
    assert filter_compliance(a)["compliant"] is None
    a["requested_filter"]={"section":"json"}
    assert filter_compliance(a)["compliant"] is False


def test_whitespace_normalized_not_semantic_labels():
    result=evaluate_query(question(),[hit("Allowed\n  only if X.")])
    assert result["hit3"] and "grounding" not in result
