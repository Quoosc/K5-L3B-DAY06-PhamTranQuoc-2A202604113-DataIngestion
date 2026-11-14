"""Regressions for frozen-design integrity and author-confirmation consistency."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    'lab6_artifact_verifier', Path(__file__).resolve().parents[2] / 'scripts/verify_artifacts.py')
verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verifier)


def write_json(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def write_csv(root, relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def frozen_design(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)
    names = ['questions.jsonl', 'protocol.md', 'coherence_positions.json', 'evidence_validation.json']
    for name in names:
        write_json(tmp_path, 'benchmark/' + name, {'snapshot': name})
    write_json(tmp_path, 'data/normalized/pages.jsonl', {'source': 'fixture'})
    write_json(tmp_path, 'artifacts/config_freeze.json', {'frozen_at': '2026-11-13T10:00:00+00:00'})
    write_json(tmp_path, 'benchmark/freeze.json', {
        'frozen_at_utc': '2026-11-13T09:00:00+00:00',
        'files': {name: verifier.sha256('benchmark/' + name) for name in names},
        'corpus_sha256': verifier.sha256('data/normalized/pages.jsonl'),
        'retrieval_results_seen_by_author': False,
    })
    return tmp_path


def test_evidence_confirmation_must_not_mutate_frozen_design(frozen_design):
    assert verifier.verify_benchmark_freeze()['frozen_files'] == 4
    write_json(frozen_design, 'benchmark/evidence_validation.json', {'human_reviewed': True})
    with pytest.raises(AssertionError, match='evidence_validation.json'):
        verifier.verify_benchmark_freeze()


def confirmation():
    return {'human_reviewed': True, 'human_label': 2, 'human_reason': 'Compared with source.',
            'human_reviewer': verifier.AUTHOR, 'human_reviewed_at': '2026-11-13T19:28:55+07:00'}


@pytest.mark.parametrize('field,value', [
    ('human_reviewed', False), ('human_reviewer', 'Ph?m Tr?n Qu?c'),
    ('human_reason', ''), ('human_label', True), ('human_label', 3),
    ('human_reviewed_at', '2026-11-13T19:28:55'),
])
def test_author_confirmation_rejects_missing_or_corrupted_evidence(field, value):
    row = confirmation()
    row[field] = value
    with pytest.raises(AssertionError):
        verifier.require_author_confirmation(row)


@pytest.fixture
def confirmed_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)
    questions = [{'id': f'q{i:02}', 'split': 'dev' if i <= 20 else 'test'} for i in range(1, 31)]
    write_json(tmp_path, 'benchmark/questions.jsonl', questions)
    write_json(tmp_path, 'benchmark/human_review.json', {
        'reviewer': verifier.AUTHOR, 'human_reviewed': True, 'reviewed_at': '2026-11-13T19:38:30+07:00',
        'source': 'benchmark/questions.jsonl', 'source_sha256': verifier.sha256('benchmark/questions.jsonl'),
        'question_count': 30, 'split_counts': {'dev': 20, 'test': 10},
        'audit_counts': {'coherence': 30, 'grounding': 30, 'metadata': 90},
    })
    positions = [{'id': f'coh{i:02}', 'document_id': 'doc', 'page': 1, 'page_offset': 0}
                 for i in range(10)]
    write_json(tmp_path, 'benchmark/coherence_positions.json', positions)
    chunks = {s: {f'c{i}': {'chunk_id': f'c{i}', 'document_id': 'doc', 'page': 1,
                           'start_offset': (i - 1) * 10, 'end_offset': i * 10}
                  for i in range(1, 4)} for s in ('A', 'B', 'C')}
    traces = [{'strategy': s, 'question_id': q['id'], 'split': 'test',
               'hits': [{'chunk_id': f'c{i}', 'rank': i} for i in range(1, 4)]}
              for s in ('A', 'B', 'C') for q in questions if q['split'] == 'test']
    coherence = [{**confirmation(), 'review_id': f'{s}-{p["id"]}', 'strategy': s,
                  'position_id': p['id'], 'chunk_id': 'c1', 'reason': 'Complete fixture paragraph.'}
                 for s in ('A', 'B', 'C') for p in positions]
    grounding = [{**confirmation(), 'review_id': f'{t["strategy"]}-{t["question_id"]}',
                  'strategy': t['strategy'], 'question_id': t['question_id'],
                  'cited_chunk_ids': ['c1'], 'reason': 'Fixture evidence covers the question.'} for t in traces]
    metadata = [{**confirmation(), 'strategy': t['strategy'], 'question_id': t['question_id'],
                 'rank': h['rank'], 'chunk_id': h['chunk_id']} for t in traces for h in t['hits']]
    experiments = [{'strategy': s, 'human_audit_confirmed': True, 'human_reviewer': verifier.AUTHOR,
                    'human_reviewed_at': '2026-11-13T19:28:55+07:00', 'human_audit_scope': 'All 50 rows.'}
                   for s in ('A', 'B', 'C')]
    write_json(tmp_path, 'results/audits/coherence_labels.json', coherence)
    write_json(tmp_path, 'results/audits/grounding_labels.json', grounding)
    write_csv(tmp_path, 'results/audits/metadata_checks.csv', metadata)
    write_csv(tmp_path, 'artifacts/experiment_log.csv', experiments)
    return tmp_path, questions, chunks, traces, experiments


def test_confirmed_audits_and_log_must_agree(confirmed_audits):
    root, questions, chunks, traces, experiments = confirmed_audits
    result = verifier.verify_author_audits(questions, chunks, traces)
    assert result['confirmed_audit_rows'] == 150
    assert result['summary']['A']['coherence_mean'] == 2
    experiments[1]['human_audit_confirmed'] = False
    write_csv(root, 'artifacts/experiment_log.csv', experiments)
    with pytest.raises(AssertionError, match='contradicts'):
        verifier.verify_author_audits(questions, chunks, traces)


def test_audit_citations_cannot_point_outside_reviewed_results(confirmed_audits):
    root, questions, chunks, traces, _ = confirmed_audits
    grounding = verifier.read_json('results/audits/grounding_labels.json')
    grounding[0]['cited_chunk_ids'] = ['unretrieved-chunk']
    write_json(root, 'results/audits/grounding_labels.json', grounding)
    with pytest.raises(AssertionError, match='outside the reviewed top-3'):
        verifier.verify_author_audits(questions, chunks, traces)


@pytest.fixture
def adjudicated_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)
    quote = 'Use Exception for user-defined exceptions.'
    page = {'text': quote, 'start_offset': 100, 'metadata': {}}
    questions = [{'id': 'q28', 'split': 'dev'}, {'id': 'qt', 'split': 'test'}]
    traces = [{'strategy': s, 'question_id': q['id'], 'split': q['split'],
               'run_at': '2026-11-13T10:00:00+00:00', 'hit1': False, 'hit3': False,
               'hits': [{'rank': 1, 'document_id': 'doc', 'page': 1,
                         'text': quote if q['id'] == 'q28' else 'Unrelated text.', 'evidence_hit': False}]}
              for q in questions for s in ('A', 'B', 'C')]
    write_json(tmp_path, 'benchmark/questions.jsonl', questions)
    for split in ('dev', 'test'):
        write_json(tmp_path, f'results/traces/{split}.jsonl', [t for t in traces if t['split'] == split])
    inputs = ['benchmark/questions.jsonl', 'results/traces/dev.jsonl', 'results/traces/test.jsonl']
    baseline, reviewed = {}, {}
    for split in ('dev', 'test'):
        baseline[split] = [{'strategy': s, 'split': split, 'n': 1, 'hit1_n': 0, 'hit3_n': 0,
                            'hit1': 0.0, 'hit3': 0.0, 'returned_chunks': 1} for s in ('A', 'B', 'C')]
        score = int(split == 'dev')
        reviewed[split] = [{**r, 'hit1_n': score, 'hit3_n': score, 'hit1': float(score), 'hit3': float(score)}
                           for r in baseline[split]]
        write_json(tmp_path, f'results/metrics/{split}.json', baseline[split])
    review = {
        'reviewer': 'Codex AI', 'method': 'Explicit post-experiment semantic review.',
        'reviewed_at': '2026-11-13T11:00:00+00:00',
        'input_sha256': {name: verifier.sha256(name) for name in inputs},
        'decisions': [{'question_id': 'q28', 'split': 'dev', 'reason': 'The alternate paragraph gives the same answer.',
                       'accepted_evidence': [{'source_id': 'doc', 'page': 1, 'evidence_quote': quote,
                                              'page_offset': 0, 'source_offset': 100}]}],
        'metrics': reviewed,
        'changes': [{'question_id': 'q28', 'strategy': s, 'split': 'dev', 'original_hit1': False,
                     'original_hit3': False, 'reviewed_hit1': True, 'reviewed_hit3': True,
                     'accepted_ranks': [1]} for s in ('A', 'B', 'C')],
        'selected_strategy': 'B', 'original_selection_preserved': True,
        'scope_note': 'Original trace and selection stay unchanged; this is a separate review.',
    }
    write_json(tmp_path, 'benchmark/human_review.json', {'post_experiment_adjudication': review})
    write_json(tmp_path, 'artifacts/selection.json', {'selected_strategy': 'B'})
    write_csv(tmp_path, 'artifacts/experiment_log.csv', [
        {'strategy': s, 'dev_hit1': 0, 'dev_hit3': 0, 'test_hit1': 0, 'test_hit3': 0,
         'dev_hit1_reviewed': 1, 'dev_hit3_reviewed': 1, 'test_hit1_reviewed': 0, 'test_hit3_reviewed': 0}
        for s in ('A', 'B', 'C')])
    return tmp_path, questions, traces, {('doc', 1): page}, review


def test_source_backed_adjudication_recomputes_all_strategies(adjudicated_fixture):
    _, questions, traces, pages, _ = adjudicated_fixture
    result = verifier.verify_adjudication(questions, traces, pages)
    assert result['changed_query_strategy_pairs'] == 3
    assert all(r['hit1'] == 1 for r in result['metrics']['dev'])
    assert all(r['hit3'] == 0 for r in result['metrics']['test'])


@pytest.mark.parametrize('mutation,error', [
    ('quote', 'claimed source position'), ('rank', 'affected trace ranks'),
    ('metric', 'metric table'), ('input', 'Adjudication input changed'),
])
def test_adjudication_rejects_wrong_quote_rank_metric_or_input(adjudicated_fixture, mutation, error):
    root, questions, traces, pages, review = adjudicated_fixture
    if mutation == 'quote':
        review['decisions'][0]['accepted_evidence'][0]['evidence_quote'] = 'Invented source quotation.'
    elif mutation == 'rank':
        review['changes'][0]['accepted_ranks'] = [2]
    elif mutation == 'metric':
        review['metrics']['dev'][0]['hit3'] = 0.5
    else:
        review['input_sha256']['results/traces/dev.jsonl'] = '0' * 64
    write_json(root, 'benchmark/human_review.json', {'post_experiment_adjudication': review})
    with pytest.raises(AssertionError, match=error):
        verifier.verify_adjudication(questions, traces, pages)
