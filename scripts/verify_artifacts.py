"""Read-only evidence checks, writing only the explicit artifact-review JSON.

This is an independent consistency check, not a human coherence/grounding audit.
Run from the repo root: python scripts/verify_artifacts.py
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHOR = 'Phạm Trấn Quốc'


def read_json(relative: str) -> dict | list:
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def read_jsonl(relative: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / relative).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def sha256(relative: str) -> str:
    h = hashlib.sha256()
    with (ROOT / relative).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def normalized(text: str) -> str:
    import unicodedata
    return ' '.join(unicodedata.normalize('NFC', text).split())


def verify_benchmark_freeze() -> dict:
    """Validate the design snapshot as well as the later configuration freeze.

    Student confirmations belong in human_review.json, not in a frozen input.
    This catches edits to evidence_validation.json that the config-only check
    did not cover.
    """
    freeze = read_json('benchmark/freeze.json')
    expected_names = {'questions.jsonl', 'protocol.md', 'coherence_positions.json',
                      'evidence_validation.json'}
    require(set(freeze['files']) == expected_names, 'Incomplete benchmark design freeze')
    for name, expected in freeze['files'].items():
        require(sha256('benchmark/' + name) == expected, f'Benchmark frozen file changed: {name}')
    require(sha256('data/normalized/pages.jsonl') == freeze['corpus_sha256'],
            'Benchmark frozen corpus changed')
    designed_at = datetime.fromisoformat(freeze['frozen_at_utc'])
    configured_at = datetime.fromisoformat(read_json('artifacts/config_freeze.json')['frozen_at'])
    require(designed_at.tzinfo is not None and designed_at < configured_at,
            'Benchmark design freeze must precede configuration freeze')
    require(freeze['retrieval_results_seen_by_author'] is False,
            'Benchmark snapshot is not marked pre-retrieval')
    return {'frozen_files': len(expected_names), 'corpus_hash_verified': True}


def require_author_confirmation(row: dict, *, csv_row: bool = False) -> int:
    reviewed = row.get('human_reviewed')
    require(reviewed == 'True' if csv_row else reviewed is True,
            'Audit row is not confirmed by the author')
    require(row.get('human_reviewer') == AUTHOR, 'Audit reviewer name is incorrect or corrupted')
    require(isinstance(row.get('human_reason'), str) and row['human_reason'].strip(),
            'Audit confirmation lacks a reason')
    reviewed_at = datetime.fromisoformat(row.get('human_reviewed_at', ''))
    require(reviewed_at.tzinfo is not None, 'Audit confirmation needs a timezone-aware date')
    label = row.get('human_label')
    if csv_row:
        require(label in ('0', '1', '2'), 'Invalid CSV human label')
        label = int(label)
    require(type(label) is int and label in (0, 1, 2), 'Invalid human audit label')
    return label


def verify_author_audits(questions: list, chunks_by_strategy: dict, all_traces: list) -> dict:
    """Check recorded author confirmations, without pretending to re-do them.

    Labels can differ from the AI proposal: summaries use human_label. Source
    IDs and cited chunks are checked independently; semantic judgments remain
    the named reviewer's responsibility.
    """
    author = read_json('benchmark/human_review.json')
    require(author.get('reviewer') == AUTHOR and author.get('human_reviewed') is True,
            'Benchmark author confirmation missing or reviewer name corrupted')
    require(datetime.fromisoformat(author['reviewed_at']).tzinfo is not None,
            'Benchmark author confirmation needs a timezone-aware date')
    require(author.get('source') == 'benchmark/questions.jsonl'
            and author.get('source_sha256') == sha256('benchmark/questions.jsonl'),
            'Author confirmation does not identify the current frozen questions')
    require(author.get('question_count') == len(questions)
            and author.get('split_counts') == {s: sum(q['split'] == s for q in questions)
                                               for s in ('dev', 'test')},
            'Author question-review scope mismatch')
    positions = read_json('benchmark/coherence_positions.json')
    position_lookup = {p['id']: p for p in positions}
    require(len(positions) == len(position_lookup) == 10, 'Expected ten unique coherence positions')
    coherence = read_json('results/audits/coherence_labels.json')
    grounding = read_json('results/audits/grounding_labels.json')
    with (ROOT / 'results/audits/metadata_checks.csv').open(encoding='utf-8', newline='') as stream:
        metadata = list(csv.DictReader(stream))
    actual_counts = {'coherence': len(coherence), 'grounding': len(grounding), 'metadata': len(metadata)}
    require(actual_counts == author.get('audit_counts') == {'coherence': 30, 'grounding': 30, 'metadata': 90},
            'Author audit counts do not match the 150 recorded judgments')
    strategies = ('A', 'B', 'C')
    expected_coherence = {(s, p) for s in strategies for p in position_lookup}
    require(len(coherence) == len({r['review_id'] for r in coherence})
            and {(r['strategy'], r['position_id']) for r in coherence} == expected_coherence,
            'Coherence audit has missing or duplicate positions')
    for row in coherence:
        require_author_confirmation(row)
        position = position_lookup[row['position_id']]
        candidates = [c for c in chunks_by_strategy[row['strategy']].values()
                      if c['document_id'] == position['document_id'] and c['page'] == position['page']
                      and c['start_offset'] <= position['page_offset'] < c['end_offset']]
        require(bool(candidates), 'Coherence position is not covered by a chunk')
        expected_chunk = sorted(candidates, key=lambda c: (-c['start_offset'], c['chunk_id']))[0]
        require(row['chunk_id'] == expected_chunk['chunk_id'], 'Coherence audit selected the wrong source chunk')
        require(bool(row.get('reason', '').strip()), 'Coherence proposal lacks a detailed rationale')
    test_traces = {(t['strategy'], t['question_id']): t for t in all_traces if t['split'] == 'test'}
    require(len(grounding) == len(test_traces) == len({r['review_id'] for r in grounding})
            and {(r['strategy'], r['question_id']) for r in grounding} == set(test_traces),
            'Grounding audit has missing or duplicate test questions')
    for row in grounding:
        require_author_confirmation(row)
        trace = test_traces[(row['strategy'], row['question_id'])]
        citations = row.get('cited_chunk_ids', [])
        require(bool(citations) and set(citations) <= {h['chunk_id'] for h in trace['hits']},
                'Grounding audit cites a chunk outside the reviewed top-3')
        require(bool(row.get('reason', '').strip()), 'Grounding proposal lacks a detailed rationale')
    expected_metadata = {(s, q, str(h['rank'])): h for (s, q), t in test_traces.items() for h in t['hits']}
    require(len(metadata) == len(expected_metadata)
            and {(r['strategy'], r['question_id'], r['rank']) for r in metadata} == set(expected_metadata),
            'Metadata confirmations have missing or duplicate ranks')
    for row in metadata:
        require(require_author_confirmation(row, csv_row=True) == 2,
                'Metadata summary claims complete/correct but an author label disagrees')
        hit = expected_metadata[(row['strategy'], row['question_id'], row['rank'])]
        require(row['chunk_id'] == hit['chunk_id'], 'Metadata confirmation references the wrong chunk')
    with (ROOT / 'artifacts/experiment_log.csv').open(encoding='utf-8', newline='') as stream:
        experiments = list(csv.DictReader(stream))
    require(len(experiments) == 3 and {r['strategy'] for r in experiments} == set(strategies),
            'Experiment log must contain exactly A, B and C')
    require(all(r.get('human_audit_confirmed') == 'True' for r in experiments),
            'Experiment log contradicts completed author confirmations')
    for row in experiments:
        require(row.get('human_reviewer') == AUTHOR and bool(row.get('human_audit_scope', '').strip()),
                'Experiment log lacks the author or audit scope')
        require(datetime.fromisoformat(row['human_reviewed_at']).tzinfo is not None,
                'Experiment audit confirmation needs a timezone-aware date')
    summary = {}
    for strategy in strategies:
        cl = [r['human_label'] for r in coherence if r['strategy'] == strategy]
        gl = [r['human_label'] for r in grounding if r['strategy'] == strategy]
        ml = [r for r in metadata if r['strategy'] == strategy]
        summary[strategy] = {'coherence_n': len(cl), 'coherence_mean': sum(cl) / len(cl),
                             'grounding_n': len(gl), 'grounding_counts': {str(i): gl.count(i) for i in (0, 1, 2)},
                             'grounding_mean': sum(gl) / len(gl),
                             'metadata_correct': len(ml), 'metadata_n': len(ml)}
    return {'author': AUTHOR, 'benchmark_questions': len(questions), 'confirmed_audit_rows': 150,
            'audit_counts': actual_counts, 'summary': summary,
            'note': 'Checks recorded confirmations and provenance; not a new human semantic audit.'}


def verify_adjudication(questions: list, all_traces: list, page_lookup: dict) -> dict:
    """Recompute a transparent post-experiment semantic-evidence supplement.

    The reviewer's semantic reason is disclosed, while source text, scope,
    ranks, counts and unchanged original selection are machine-verifiable.
    Original questions, traces and metrics are never rewritten here.
    """
    review = read_json('benchmark/human_review.json')['post_experiment_adjudication']
    require(review.get('reviewer') == 'Codex AI' and bool(review.get('method', '').strip()),
            'Post-experiment adjudication must disclose its AI reviewer and method')
    expected_inputs = {'benchmark/questions.jsonl', 'results/traces/dev.jsonl', 'results/traces/test.jsonl'}
    require(set(review['input_sha256']) == expected_inputs, 'Incomplete adjudication input manifest')
    for name, expected in review['input_sha256'].items():
        require(sha256(name) == expected, f'Adjudication input changed: {name}')
    reviewed_at = datetime.fromisoformat(review['reviewed_at'])
    last_test = max(datetime.fromisoformat(t['run_at']) for t in all_traces if t['split'] == 'test')
    require(reviewed_at.tzinfo is not None and reviewed_at > last_test,
            'Adjudication must be disclosed as after the original test run')
    q_lookup = {q['id']: q for q in questions}
    decisions = review['decisions']
    decision_lookup = {d['question_id']: d for d in decisions}
    require(len(decision_lookup) == len(decisions), 'Duplicate adjudication decision')
    for decision in decisions:
        q = q_lookup[decision['question_id']]
        require(decision['split'] == q['split'] and bool(decision.get('reason', '').strip()),
                'Adjudication question/split/rationale mismatch')
        require(bool(decision['accepted_evidence']), 'Accepted decision lacks source evidence')
        for evidence in decision['accepted_evidence']:
            page = page_lookup[(evidence['source_id'], evidence['page'])]
            quote, offset = evidence['evidence_quote'], evidence['page_offset']
            require(bool(quote.strip()) and page['text'][offset:offset + len(quote)] == quote,
                    'Adjudicated evidence is not at its claimed source position')
            require(evidence['source_offset'] == page['start_offset'] + offset,
                    'Adjudicated evidence source offset mismatch')
            require(all(page['metadata'].get(k) == v for k, v in q.get('filter', {}).items()),
                    'Adjudicated evidence violates the question scope')
    reviewed_traces = []
    changes = []
    for trace in all_traces:
        decision = decision_lookup.get(trace['question_id'], {})
        alternatives = decision.get('accepted_evidence', [])
        accepted_ranks = [h['rank'] for h in trace['hits'] if any(
            h['document_id'] == g['source_id'] and h['page'] == g['page']
            and normalized(g['evidence_quote']) in normalized(h['text']) for g in alternatives)]
        hits = [h['evidence_hit'] or h['rank'] in accepted_ranks for h in trace['hits']]
        hit1, hit3 = bool(hits and hits[0]), any(hits[:3])
        reviewed_traces.append({**trace, 'hit1': hit1, 'hit3': hit3})
        if (hit1, hit3) != (trace['hit1'], trace['hit3']):
            changes.append({'question_id': trace['question_id'], 'strategy': trace['strategy'],
                            'split': trace['split'], 'original_hit1': trace['hit1'],
                            'original_hit3': trace['hit3'], 'reviewed_hit1': hit1,
                            'reviewed_hit3': hit3, 'accepted_ranks': accepted_ranks})
    metrics = {}
    for split in ('dev', 'test'):
        metrics[split] = []
        for strategy in ('A', 'B', 'C'):
            subset = [t for t in reviewed_traces if t['split'] == split and t['strategy'] == strategy]
            n = len(subset)
            require(n > 0, 'Missing adjudication split/strategy')
            h1, h3 = sum(t['hit1'] for t in subset), sum(t['hit3'] for t in subset)
            metrics[split].append({'strategy': strategy, 'split': split, 'n': n,
                                   'hit1_n': h1, 'hit3_n': h3, 'hit1': h1 / n, 'hit3': h3 / n,
                                   'returned_chunks': sum(len(t['hits']) for t in subset)})
    require(metrics == review['metrics'], 'Adjudicated metric table differs from source-backed recomputation')
    sort_key = lambda row: (row['split'], row['question_id'], row['strategy'])
    require(sorted(changes, key=sort_key) == sorted(review['changes'], key=sort_key),
            'Adjudication change log does not match the affected trace ranks')
    selection = read_json('artifacts/selection.json')
    require(review.get('original_selection_preserved') is True
            and review.get('selected_strategy') == selection['selected_strategy'],
            'Adjudication changes or misstates the original dev selection')
    require(bool(review.get('scope_note', '').strip()), 'Adjudication needs an explicit scope/limitation note')
    with (ROOT / 'artifacts/experiment_log.csv').open(encoding='utf-8', newline='') as stream:
        experiments = {r['strategy']: r for r in csv.DictReader(stream)}
    for split, rows in metrics.items():
        baseline = {r['strategy']: r for r in read_json(f'results/metrics/{split}.json')}
        for row in rows:
            experiment = experiments[row['strategy']]
            for metric in ('hit1', 'hit3'):
                require(math.isclose(float(experiment[f'{split}_{metric}_reviewed']), row[metric]),
                        'Experiment log reviewed metrics disagree with adjudication')
                require(math.isclose(float(experiment[f'{split}_{metric}']), baseline[row['strategy']][metric]),
                        'Experiment log original metrics disagree with frozen run')
    return {'reviewer': review['reviewer'], 'decisions': len(decisions), 'changed_query_strategy_pairs': len(changes),
            'metrics': metrics, 'original_selection_preserved': True,
            'note': 'Checks source spans and recomputed adjustments; semantic acceptance remains a disclosed AI review.'}


def verify() -> dict:
    checks: list[dict] = []
    report = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'Codex AI', 'human_reviewed': False,
        'method': 'Independent exact source/span/hash, tokenizer, trace, metric and reproduction checks; '
                  'not a human semantic audit or grade guarantee.',
        'command': '.venv/Scripts/python.exe scripts/verify_artifacts.py',
        'checks': checks,
    }

    def passed(name: str, **details) -> None:
        checks.append({'name': name, 'pass': True, **details})

    try:
        passed('benchmark_design_freeze_hashes', **verify_benchmark_freeze())
        freeze = read_json('artifacts/config_freeze.json')
        for name, expected in freeze['fingerprints'].items():
            require(sha256(name) == expected, f'Frozen file changed: {name}')
        require(sha256('results/traces/dev.jsonl') == freeze['dev_trace_sha256'], 'Dev trace changed after freeze')
        for split in ('dev', 'test'):
            require(read_json(f'results/metrics/{split}_context.json')['fingerprints'] == freeze['fingerprints'],
                    f'{split} provenance does not match freeze')
        passed('frozen_code_input_and_trace_hashes', frozen_files=len(freeze['fingerprints']))

        manifest = read_json('data/manifest.json')
        pages = read_jsonl('data/normalized/pages.jsonl')
        page_lookup = {(p['document_id'], p['page']): p for p in pages}
        require(len(page_lookup) == len(pages) == 40, 'Expected 40 unique source units')
        require(sha256('data/normalized/pages.jsonl') == manifest['pages_sha256'], 'Corpus page hash mismatch')
        selected = 0
        for source in manifest['sources']:
            require(sha256(source['raw_path']) == source['raw_sha256'], 'Raw source hash mismatch')
            require(sha256(source['normalized_path']) == source['normalized_sha256'], 'Normalized source hash mismatch')
            document = (ROOT / source['normalized_path']).read_text(encoding='utf-8')
            for unit in source['units']:
                if not unit['selected']:
                    continue
                selected += 1
                page = page_lookup[(source['document_id'], unit['page'])]
                require(page['text'] == document[unit['start_offset']:unit['end_offset']], 'Source-unit span mismatch')
                require(len(page['text']) == unit['character_count'] >= 1500, 'Invalid content-unit count')
                require(page['source'] == source['source_path_or_url'], 'Source URL mismatch')
                require(all(page['metadata'][k] == v for k, v in source['metadata'].items()), 'Source metadata mismatch')
        require(selected == len(pages), 'Selected source-unit count mismatch')
        passed('corpus_provenance', documents=len(manifest['sources']), source_units=len(pages),
               total_characters=sum(len(p['text']) for p in pages),
               minimum_unit_characters=min(len(p['text']) for p in pages))

        chunks_by_strategy: dict[str, dict] = {}
        chunk_counts = {}
        for strategy in ('A', 'B'):
            chunks = read_jsonl(f'results/chunks/{strategy}.jsonl')
            chunks_by_strategy[strategy] = {c['chunk_id']: c for c in chunks}
            require(len(chunks_by_strategy[strategy]) == len(chunks), 'Duplicate chunk IDs')
            coverage = {key: set() for key in page_lookup}
            for chunk in chunks:
                key = (chunk['document_id'], chunk['page'])
                page = page_lookup[key]
                require(chunk['source'] == page['source'] and chunk['metadata'] == page['metadata'], 'Chunk source/metadata mismatch')
                require(chunk['text'] == page['text'][chunk['start_offset']:chunk['end_offset']], 'Chunk span mismatch')
                coverage[key].update(range(chunk['start_offset'], chunk['end_offset']))
            require(all(all(i in coverage[key] for i, char in enumerate(page['text']) if not char.isspace())
                        for key, page in page_lookup.items()), 'Non-whitespace source content missing from chunks')
            chunk_counts[strategy] = len(chunks)
        chunks_by_strategy['C'] = chunks_by_strategy['B']
        passed('chunk_identity_provenance_and_source_coverage', counts=chunk_counts,
               C_reuses_B=True, coverage_scope='all non-whitespace characters of all selected source units')

        os.environ['HF_HUB_OFFLINE'] = '1'
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / '.local/model-snapshot'), local_files_only=True)
        token_summary = {}
        for strategy in ('A', 'B'):
            content_lengths = []
            full_lengths = []
            for chunk in chunks_by_strategy[strategy].values():
                content = len(tokenizer.encode(chunk['text'], add_special_tokens=False, truncation=False))
                full = len(tokenizer.encode('passage: ' + chunk['text'], add_special_tokens=True, truncation=False))
                require(content == chunk['token_count'] <= 320 and full <= 512, 'Embedding document token budget exceeded')
                content_lengths.append(content)
                full_lengths.append(full)
            token_summary[strategy] = {'max_content_tokens': max(content_lengths), 'max_full_input_tokens': max(full_lengths)}
        questions = read_jsonl('benchmark/questions.jsonl')
        query_max = max(len(tokenizer.encode('query: ' + q['question'], add_special_tokens=True, truncation=False))
                        for q in questions)
        require(query_max <= 512, 'Embedding query token budget exceeded')
        passed('actual_tokenizer_input_limits', strategies=token_summary, max_query_input_tokens=query_max,
               model_limit=512)

        q_lookup = {q['id']: q for q in questions}
        all_traces = []
        metric_summary = {}
        for split, expected_n in (('dev', 20), ('test', 10)):
            traces = read_jsonl(f'results/traces/{split}.jsonl')
            metrics = read_json(f'results/metrics/{split}.json')
            all_traces.extend(traces)
            require(len(traces) == expected_n * 3, 'Trace cardinality mismatch')
            metric_summary[split] = {}
            for strategy in ('A', 'B', 'C'):
                subset = [t for t in traces if t['strategy'] == strategy]
                require(len(subset) == len({t['question_id'] for t in subset}) == expected_n, 'Trace query count mismatch')
                for trace in subset:
                    q = q_lookup[trace['question_id']]
                    require(trace['question'] == q['question'] and trace['split'] == q['split'], 'Trace question/split mismatch')
                    require(trace['requested_filter'] == q.get('filter', {}), 'Requested filter mismatch')
                    require(trace['applied_filter'] == (q.get('filter', {}) if strategy == 'C' else {}), 'Applied filter mismatch')
                    require(len(trace['hits']) == 3, 'Expected three results per actual run query')
                    for hit in trace['hits']:
                        chunk = chunks_by_strategy[strategy][hit['chunk_id']]
                        require(all(hit[k] == v for k, v in chunk.items()), 'Trace differs from saved chunk')
                        gold_matches = [g for g in trace['gold'] if g['source_id'] == hit['document_id']
                                        and g['page'] == hit['page'] and normalized(g['evidence_quote'])
                                        and normalized(g['evidence_quote']) in normalized(hit['text'])]
                        require(hit['evidence_hit'] == bool(gold_matches), 'Evidence-hit mismatch')
                        if strategy == 'C':
                            require(all(hit['metadata'].get(k) == v for k, v in trace['requested_filter'].items()), 'Filter violation')
                    require(trace['hit1'] == bool(trace['hits'][0]['evidence_hit']), 'Hit@1 flag mismatch')
                    require(trace['hit3'] == any(h['evidence_hit'] for h in trace['hits']), 'Hit@3 flag mismatch')
                    require(all(trace['hits'][i]['score'] >= trace['hits'][i+1]['score'] for i in range(2)), 'Scores not descending')
                result = next(m for m in metrics if m['strategy'] == strategy and m['split'] == split)
                h1, h3 = sum(t['hit1'] for t in subset), sum(t['hit3'] for t in subset)
                require((result['n'], result['hit1_n'], result['hit3_n']) == (expected_n, h1, h3), 'Metric numerator/denominator mismatch')
                require(math.isclose(result['hit1'], h1 / expected_n) and math.isclose(result['hit3'], h3 / expected_n), 'Metric rate mismatch')
                metric_summary[split][strategy] = {'n': expected_n, 'hit1_n': h1, 'hit3_n': h3}
            for q in questions:
                if q['split'] != split or q.get('filter'):
                    continue
                b = next(t for t in traces if t['question_id'] == q['id'] and t['strategy'] == 'B')
                c = next(t for t in traces if t['question_id'] == q['id'] and t['strategy'] == 'C')
                require([h['chunk_id'] for h in b['hits']] == [h['chunk_id'] for h in c['hits']], 'B/C no-filter invariant mismatch')
        passed('trace_metrics_evidence_filters_and_BC_invariant', traces=len(all_traces), metrics=metric_summary)

        last_dev = max(t['run_at'] for t in all_traces if t['split'] == 'dev')
        first_test = min(t['run_at'] for t in all_traces if t['split'] == 'test')
        require(datetime.fromisoformat(last_dev) < datetime.fromisoformat(freeze['frozen_at']) < datetime.fromisoformat(first_test),
                'Dev/freeze/test chronology violated')
        selection = read_json('artifacts/selection.json')
        require(selection['selected_strategy'] == freeze['selected_strategy'] == 'B', 'Selection mismatch')
        require(selection['selection_split'] == 'dev', 'Selection did not use dev')
        passed('freeze_chronology', last_dev=last_dev, frozen_at=freeze['frozen_at'], first_test=first_test,
               selected_strategy='B')

        with (ROOT / 'results/audits/metadata_checks.csv').open(encoding='utf-8', newline='') as stream:
            metadata_rows = list(csv.DictReader(stream))
        lookup = {(t['question_id'], t['strategy'], str(h['rank'])): h for t in all_traces if t['split'] == 'test' for h in t['hits']}
        require(len(metadata_rows) == len(lookup) == 90, 'Metadata audit cardinality mismatch')
        keys = {(r['question_id'], r['strategy'], r['rank']) for r in metadata_rows}
        require(keys == lookup.keys(), 'Metadata audit has missing or duplicate trace ranks')
        student_rows = 0
        for row in metadata_rows:
            hit = lookup[(row['question_id'], row['strategy'], row['rank'])]
            page = page_lookup[(hit['document_id'], hit['page'])]
            require(row['chunk_id'] == hit['chunk_id'] and row['source'] == hit['source'] == page['source'], 'Metadata source mismatch')
            require(hit['metadata'] == page['metadata'], 'Metadata payload mismatch')
            require(hit['text'] == page['text'][hit['start_offset']:hit['end_offset']], 'Metadata text span mismatch')
            require(all(row[k] == 'True' for k in ('source_correct', 'metadata_correct', 'span_correct', 'automated_complete_correct')),
                    'CSV automated flags contradict exact comparison')
            review_status = row['human_reviewed'].strip().lower()
            require(review_status in ('true', 'false'), 'Invalid human_reviewed value')
            if review_status == 'true':
                require(all(row.get(k, '').strip() for k in
                            ('human_label', 'human_reason', 'human_reviewer', 'human_reviewed_at')),
                        'Confirmed metadata row needs label, reason, reviewer and date')
                datetime.fromisoformat(row['human_reviewed_at'])
                student_rows += 1
        passed('independent_metadata_crosscheck', correct=90, checked=90, student_rows_recorded=student_rows)

        passed('author_confirmations_and_audit_labels', **verify_author_audits(questions, chunks_by_strategy, all_traces))
        passed('post_experiment_evidence_adjudication', **verify_adjudication(questions, all_traces, page_lookup))

        checks_storage = read_json('results/storage_checks.json')
        require(checks_storage['pass'] and checks_storage['first_count'] == checks_storage['repeat_count'] == 2
                and checks_storage['after_update_count'] == 1 and checks_storage['after_removal_count'] == 0
                and checks_storage['before_reopen'] == checks_storage['after_reopen'] == chunk_counts, 'Storage evidence mismatch')
        passed('recorded_storage_test_consistency', note='Checks recorded real-run output; does not mutate the database.')

        model_manifest = read_json('artifacts/model_download.json')
        for entry in model_manifest['files']:
            name = model_manifest['local_path'] + '/' + entry['file']
            require((ROOT / name).stat().st_size == entry['bytes'] and sha256(name) == entry['sha256'], 'Model asset hash mismatch')
        passed('local_model_asset_integrity', assets=len(model_manifest['files']), revision=model_manifest['revision'])

        require((ROOT / 'results/reproduction/traces/test.jsonl').exists(), 'Reproduction test artifacts are not ready')
        chunk_bytes_match = True
        for strategy in ('A', 'B'):
            original_path = f'results/chunks/{strategy}.jsonl'
            reproduced_path = f'results/reproduction/chunks/{strategy}.jsonl'
            require(read_jsonl(original_path) == read_jsonl(reproduced_path), 'Reproduced chunk records differ')
            chunk_bytes_match &= sha256(original_path) == sha256(reproduced_path)
        for split in ('dev', 'test'):
            require(read_json(f'results/metrics/{split}.json') == read_json(f'results/reproduction/metrics/{split}.json'),
                    'Reproduced metric table differs')
            original = {(t['question_id'], t['strategy']): t for t in read_jsonl(f'results/traces/{split}.jsonl')}
            repeated = {(t['question_id'], t['strategy']): t for t in read_jsonl(f'results/reproduction/traces/{split}.jsonl')}
            require(original.keys() == repeated.keys(), 'Reproduced trace identities differ')
            for key, one in original.items():
                two = repeated[key]
                require((one['hit1'], one['hit3']) == (two['hit1'], two['hit3']), 'Reproduced hit labels differ')
                require([h['chunk_id'] for h in one['hits']] == [h['chunk_id'] for h in two['hits']], 'Reproduced rankings differ')
                require(all(math.isclose(a['score'], b['score'], abs_tol=1e-6, rel_tol=0)
                            for a, b in zip(one['hits'], two['hits'])), 'Reproduced scores differ beyond tolerance')
        passed('database_reproduction_parity', chunk_records_identical=True,
               chunk_hashes_identical=chunk_bytes_match, traces_compared=90,
               metric_tables_identical=True, score_absolute_tolerance=1e-6)
        report['pass'] = True
    except Exception as exc:
        report['pass'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['limitations'] = [
        'Validates the recorded student confirmations; it cannot independently prove or replace the human review.',
        'Exact evidence matching is not semantic adjudication of all possible alternative sources.',
        'Recorded timestamps and hashes establish internal consistency, not externally signed provenance.',
        'No model inference or database mutation is performed by this reviewer script.',
    ]
    return report


if __name__ == '__main__':
    review = verify()
    destination = ROOT / 'results/artifact_review.json'
    destination.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pass': review['pass'], 'checks': len(review['checks']), 'error': review.get('error'),
                      'report': str(destination)}, ensure_ascii=False))
    raise SystemExit(0 if review['pass'] else 1)
