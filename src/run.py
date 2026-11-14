"""Reproducible Lab6 CLI. Run from the repository root: python -m src.run --help."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import evaluate_query, summarize, golds, normalize_evidence, filter_compliance

ROOT = Path(__file__).resolve().parents[1]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_fingerprints():
    names = ["configs/strategies.json", "benchmark/questions.jsonl", "benchmark/protocol.md",
             "benchmark/coherence_positions.json", "data/normalized/pages.jsonl"]
    names += ["src/"+name+".py" for name in ("run", "evaluation", "chunking", "embeddings", "store", "ingestion")]
    return {name: digest(ROOT/name) for name in names}


def load_model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import random
    import numpy as np
    import torch
    from .embeddings import LocalEmbedder
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    cfg = read_json(ROOT/"configs/strategies.json")
    if cfg["revision"] == "main":
        raise ValueError("Pin the model commit SHA in configs/strategies.json before the real experiment.")
    download=read_json(ROOT/"artifacts/model_download.json")
    if download["revision"]!=cfg["revision"]: raise ValueError("Downloaded model revision does not match config")
    for asset in download["files"]:
        if digest(ROOT/".local/model-snapshot"/asset["file"])!=asset["sha256"]:
            raise ValueError("Model snapshot checksum mismatch: "+asset["file"])
    return LocalEmbedder(model_name=cfg["model"], revision=cfg["revision"], model_path=str(ROOT/".local/model-snapshot"),
                         cache_folder=str(ROOT/".local/model"), batch_size=cfg["batch_size"], device=cfg["device"])


def validate():
    pages = read_jsonl(ROOT/"data/normalized/pages.jsonl")
    questions = read_jsonl(ROOT/"benchmark/questions.jsonl")
    errors = []
    design=read_json(ROOT/"benchmark/freeze.json")
    if design["corpus_sha256"]!=digest(ROOT/"data/normalized/pages.jsonl"): errors.append("corpus differs from benchmark design freeze")
    for name,expected in design["files"].items():
        if digest(ROOT/"benchmark"/name)!=expected: errors.append("benchmark freeze mismatch: "+name)
    by_page = {(p["document_id"], p["page"]): p for p in pages}
    if len(pages) < 30: errors.append("fewer than 30 source units")
    if len(by_page) != len(pages): errors.append("duplicate source unit IDs")
    if any(len(p["text"]) < 1500 for p in pages): errors.append("source unit shorter than1500characters")
    if len({digest_text(p["text"]) for p in pages}) != len(pages): errors.append("duplicated unit text")
    counts = {split: sum(q["split"] == split for q in questions) for split in ("dev", "test")}
    if counts["dev"] < 15 or counts["test"] < 5: errors.append("insufficient split counts")
    if len({q["id"] for q in questions}) != len(questions): errors.append("duplicate question IDs")
    filtered = [q for q in questions if q.get("filter")]
    if len(filtered) < 3 or not any(q["split"] == "test" for q in filtered): errors.append("insufficient filter questions")
    evidence_rows = []
    for q in questions:
        if q.get("split") not in ("dev","test"): errors.append("unknown split: "+q["id"])
        if not q.get("question", "").strip() or not q.get("expected_answer", "").strip(): errors.append("empty question/answer: "+q["id"])
        if set(q.get("filter",{})) - {"section","document_type","language","version"}: errors.append("unsupported filter: "+q["id"])
        for g in golds(q):
            source = g.get("source_id", g.get("gold_source_id")); page = g.get("page", g.get("gold_page"))
            p = by_page.get((source, page))
            quote = normalize_evidence(g["evidence_quote"])
            valid = bool(p and quote and quote in normalize_evidence(p["text"]))
            filter_valid = bool(p) and all(p["metadata"].get(k) == v for k,v in q.get("filter",{}).items())
            evidence_rows.append({"question_id":q["id"],"source_id":source,"page":page,"excerpt_exists":valid,"filter_consistent":filter_valid})
            if not valid or not filter_valid: errors.append(f"invalid evidence/filter: {q['id']} {source}:{page}")
    result = {"checked_at":utcnow(),"pages":len(pages),"questions":len(questions),"splits":counts,
              "filter_questions":len(filtered),"filter_test":sum(q["split"]=="test" for q in filtered),
              "evidence":evidence_rows,"errors":errors,"pass":not errors,
              "method":"automated excerpt existence/source/filter verification; semantic/human review recorded separately"}
    write_json(ROOT/"results/input_validation.json",result)
    if errors: raise ValueError("; ".join(errors))
    print(json.dumps({k:v for k,v in result.items() if k not in ("evidence",)},ensure_ascii=False))
    return result


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest(output, db):
    import numpy as np
    from .chunking import chunk_pages
    from .store import QdrantStore
    validate()
    cfg = read_json(ROOT/"configs/strategies.json")
    pages = read_jsonl(ROOT/"data/normalized/pages.jsonl")
    started = time.perf_counter(); model = load_model(); load_sec = time.perf_counter()-started
    store = QdrantStore(str(db), dimensions=model.dimensions)
    stats = {}; built = {}
    try:
        for strategy in ("A","B"):
            c = cfg["strategies"][strategy]
            start = time.perf_counter()
            chunks = chunk_pages(pages,strategy,model.tokenizer,max_tokens=c["max_tokens"],overlap=c["overlap"])
            chunk_sec = time.perf_counter()-start
            texts = [c["text"] for c in chunks]
            cache_key = digest_text(json.dumps({"model":cfg["model"],"revision":cfg["revision"],"texts":texts,
                "adapter_sha256":digest(ROOT/"src/embeddings.py"),
                "embedding_policy":{k:model.metadata[k] for k in ("document_prefix","query_prefix","normalize_embeddings","max_input_tokens","dimensions")}},ensure_ascii=False,sort_keys=True))
            cache = ROOT/".local/embeddings"/(cache_key+".npy"); cache.parent.mkdir(parents=True,exist_ok=True)
            warm = cache.exists(); start=time.perf_counter()
            if warm: vectors=np.load(cache,allow_pickle=False)
            else:
                vectors=np.asarray(model.embed_documents(texts),dtype=np.float32)
                np.save(cache,vectors,allow_pickle=False)
            embed_sec=time.perf_counter()-start
            start=time.perf_counter(); sync=store.sync("lab6_"+strategy,chunks,vectors.tolist()); store_sec=time.perf_counter()-start
            write_jsonl(output/"chunks"/(strategy+".jsonl"),chunks)
            lengths=[len(model.tokenizer.encode(t,add_special_tokens=False)) for t in texts]
            stats[strategy]={"chunks":len(chunks),"tokens_total":sum(lengths),"chars_total":sum(map(len,texts)),
                             "token_min":min(lengths),"token_max":max(lengths),"token_mean":statistics.mean(lengths),
                             "token_p50":float(np.percentile(lengths,50)),"token_p95":float(np.percentile(lengths,95)),
                             "chunk_seconds":chunk_sec,"embedding_seconds":embed_sec,"store_seconds":store_sec,
                             "embedding_cache": "warm" if warm else "cold","embedding_cache_key":cache_key,
                             "chunk_sha256":digest(output/"chunks"/(strategy+".jsonl")),"sync":sync,
                             "api_cost_usd":0,"cost_note":"local inference; hardware/electricity not monetized"}
            built[strategy]=(chunks,vectors.tolist())
            print(f"Ingested {strategy}: {len(chunks)} chunks, cache={warm}",flush=True)
        stats["C"]={**stats["B"],"reuses":"B","additional_index_seconds":0,
                    "note":"same B collection/chunks/vectors; B build cost applies, filter is only intervention"}
        probe=built["A"][0][:2]; vec=built["A"][1][:2]
        store.sync("lab6_storage_probe",probe,vec); count1=store.count("lab6_storage_probe")
        store.sync("lab6_storage_probe",probe,vec); count2=store.count("lab6_storage_probe")
        changed=[{**probe[0],"chunk_id":probe[0]["chunk_id"]+"_changed","text":probe[0]["text"]+"\n[storage test update]"}]
        store.sync("lab6_storage_probe",changed,vec[:1]); count3=store.count("lab6_storage_probe")
        probe_hits=store.search("lab6_storage_probe",vec[0],top_k=3)
        updated_ids=[h["chunk_id"] for h in probe_hits]
        store.sync("lab6_storage_probe",[],[]); count4=store.count("lab6_storage_probe")
        before={s:store.count("lab6_"+s) for s in ("A","B")}
    finally:
        store.close()
    reopened=QdrantStore(str(db),dimensions=model.dimensions)
    try: after={s:reopened.count("lab6_"+s) for s in ("A","B")}
    finally: reopened.close()
    checks={"checked_at":utcnow(),"first_count":count1,"repeat_count":count2,"after_update_count":count3,
            "updated_ids":updated_ids,"after_removal_count":count4,"before_reopen":before,"after_reopen":after,
            "pass":count1==count2==2 and count3==1 and updated_ids==[changed[0]["chunk_id"]] and count4==0 and before==after,
            "note":"dedicated probe collection using real embedding vectors; not included in benchmark"}
    write_json(output/"storage_checks.json",checks)
    if not checks["pass"]: raise AssertionError("storage checks failed")
    write_json(output/"resource_stats.json",{"created_at":utcnow(),"model_load_seconds":load_sec,"strategies":stats})
    versions={name:importlib.metadata.version(name) for name in ("torch","sentence-transformers","transformers","qdrant-client","numpy","tokenizers","huggingface-hub")}
    write_json(output/"run_manifest.json",{"created_at":utcnow(),"python":platform.python_version(),"platform":platform.platform(),
                "versions":versions,"model":model.metadata,"config":cfg,"fingerprints":input_fingerprints(),
                "vector_store":{"backend":"Qdrant local persistent","path":str(db.relative_to(ROOT)) if db.is_relative_to(ROOT) else str(db),"metric":"cosine"},
                "api_cost_usd":0,"random_seed":42,"note":"No API generation; deterministic fixed corpus and explicit strategies."})


def benchmark(split, output, db):
    import numpy as np
    from .store import QdrantStore
    if split=="test":
        frozen=read_json(ROOT/"artifacts/config_freeze.json")
        if frozen["fingerprints"]!=input_fingerprints(): raise ValueError("Inputs/code changed since freeze; cannot claim original holdout.")
        if not (ROOT/"artifacts/selection.json").exists(): raise ValueError("Dev selection required before test")
    cfg=read_json(ROOT/"configs/strategies.json")
    manifest=read_json(output/"run_manifest.json")
    if manifest["fingerprints"]!=input_fingerprints(): raise ValueError("Re-ingest after changing source/config/runtime")
    if (ROOT/manifest["vector_store"]["path"]).resolve()!=db: raise ValueError("Database path does not match the ingest manifest")
    questions=[q for q in read_jsonl(ROOT/"benchmark/questions.jsonl") if q["split"]==split]
    model=load_model(); store=QdrantStore(str(db),dimensions=model.dimensions)
    traces=[]; times={s:[] for s in ("A","B","C")}; query_embed=[]
    try:
        for s in ("A","B"):
            if store.count("lab6_"+s)!=len(read_jsonl(output/"chunks"/(s+".jsonl"))): raise ValueError("Stored collection count differs from ingest output")
        for q in questions:
            start=time.perf_counter(); vector=model.embed_query(q["question"]); query_embed.append(time.perf_counter()-start)
            # Query embedding is identical across all three interventions.
            for strategy in ("A","B","C"):
                collection="lab6_"+("B" if strategy=="C" else strategy)
                applied=q.get("filter",{}) if strategy=="C" else {}
                start=time.perf_counter(); hits=store.search(collection,vector,top_k=3,filters=applied or None); elapsed=time.perf_counter()-start
                times[strategy].append(elapsed)
                row=evaluate_query(q,hits)
                row.update({"strategy":strategy,"applied_filter":applied,"search_seconds":elapsed,"query_embedding_seconds":query_embed[-1],
                            "run_at":utcnow(),"model_revision":cfg["revision"],"config_sha256":digest(ROOT/"configs/strategies.json")})
                row["filter_compliance"]=filter_compliance(row); traces.append(row)
            print(f"Retrieved {split} {q['id']} for A/B/C",flush=True)
    finally: store.close()
    write_jsonl(output/"traces"/(split+".jsonl"),traces)
    metrics=summarize(traces); write_json(output/"metrics"/(split+".json"),metrics)
    write_json(output/"metrics"/(split+"_context.json"),{"completed_at":utcnow(),"fingerprints":input_fingerprints(),
               "trace_sha256":digest(output/"traces"/(split+".jsonl")),"metrics_sha256":digest(output/"metrics"/(split+".json"))})
    write_json(output/"metrics"/(split+"_timing.json"),{"created_at":utcnow(),"query_embedding_total":sum(query_embed),
                  "search":{s:{"total_seconds":sum(v),"p50_seconds":float(np.percentile(v,50)),"p95_seconds":float(np.percentile(v,95))} for s,v in times.items()},
                  "note":"single sequential run; query embeddings reused; timings hardware/cache dependent"})
    # Invariant: C must not change results for a question without a filter.
    for q in questions:
        if not q.get("filter"):
            b=next(t for t in traces if t["question_id"]==q["id"] and t["strategy"]=="B")
            c=next(t for t in traces if t["question_id"]==q["id"] and t["strategy"]=="C")
            if [h["chunk_id"] for h in b["hits"]]!=[h["chunk_id"] for h in c["hits"]]: raise AssertionError("B/C no-filter invariant failed")
    print(json.dumps(metrics,indent=2))


def freeze(output):
    if (ROOT/"results/traces/test.jsonl").exists(): raise ValueError("Original test already evaluated; do not rewrite its freeze record")
    context=read_json(output/"metrics/dev_context.json")
    if context["fingerprints"]!=input_fingerprints(): raise ValueError("Dev results are stale for current inputs/runtime")
    if context["trace_sha256"]!=digest(output/"traces/dev.jsonl") or context["metrics_sha256"]!=digest(output/"metrics/dev.json"):
        raise ValueError("Dev result integrity mismatch")
    rows=read_json(output/"metrics/dev.json"); timings=read_json(output/"metrics/dev_timing.json")
    # Recorded in benchmark/protocol.md before any benchmark retrieval.
    chosen=sorted(rows,key=lambda r:(-r["hit3"],-r["hit1"],timings["search"][r["strategy"]]["p95_seconds"],{"B":0,"A":1,"C":2}[r["strategy"]]))[0]
    selection={"created_at":utcnow(),"selected_strategy":chosen["strategy"],"selection_split":"dev",
               "rule":"max dev Hit@3, max dev Hit@1, min dev p95 search latency, final tie B/A/C",
               "dev_metrics":rows,"timing":timings,"note":"Single-run latency is a noisy tie-break; selection is not a universal ranking."}
    write_json(ROOT/"artifacts/selection.json",selection)
    write_json(ROOT/"artifacts/config_freeze.json",{"frozen_at":utcnow(),"fingerprints":input_fingerprints(),
                "selected_strategy":chosen["strategy"],"dev_trace_sha256":digest(output/"traces/dev.jsonl"),"test_results_seen":False})
    print("Frozen before test. Selected strategy:",chosen["strategy"])


def export_audit(output):
    import random
    traces=read_jsonl(output/"traces/test.jsonl")
    # Partially blinded: deterministic shuffled order; no strategy/score/hit label.
    random.Random(42).shuffle(traces)
    rows=[]
    for i,t in enumerate(traces,1):
        rows.append({"review_id":f"G{i:03}","question_id":t["question_id"],"question":t["question"],
                     "expected_answer":t["expected_answer"],"gold":t["gold"],"filter":t["requested_filter"],
                     "hits":[{k:v for k,v in h.items() if k not in ("evidence_hit","matched_gold","chunk_policy","score")} for h in t["hits"]]})
    write_jsonl(output/"audits/grounding_review_input.jsonl",rows)
    write_json(output/"audits/grounding_key.json",{f"G{i:03}":{"question_id":t["question_id"],"strategy":t["strategy"]} for i,t in enumerate(traces,1)})
    print("Grounding review input exported; labels are not inferred by this command.")


def query(args,db):
    from .store import QdrantStore
    model=load_model(); store=QdrantStore(str(db),dimensions=model.dimensions)
    try:
        hits=store.search("lab6_"+("B" if args.strategy=="C" else args.strategy),model.embed_query(args.text),top_k=3,
                          filters=json.loads(args.filters) if args.strategy=="C" and args.filters else None)
        print(json.dumps(hits,ensure_ascii=False,indent=2))
    finally: store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["validate","ingest","dev","freeze","test","audit-export","query"])
    parser.add_argument("--output",default="results"); parser.add_argument("--db",default=".local/qdrant")
    parser.add_argument("--strategy",choices=["A","B","C"],default="C")
    parser.add_argument("--text",default="How can JSON object keys be sorted during serialization?")
    parser.add_argument("--filters",default="")
    args=parser.parse_args(); output=(ROOT/args.output).resolve(); db=(ROOT/args.db).resolve()
    if not output.is_relative_to(ROOT) or not db.is_relative_to(ROOT): raise ValueError("Output/database paths must stay inside this repository")
    if args.command=="validate": validate()
    elif args.command=="ingest": ingest(output,db)
    elif args.command in ("dev","test"): benchmark(args.command,output,db)
    elif args.command=="freeze": freeze(output)
    elif args.command=="audit-export": export_audit(output)
    else: query(args,db)


if __name__=="__main__": main()
