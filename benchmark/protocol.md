# Benchmark protocol — fixed before retrieval

This protocol describes the evaluation design, not results. Questions and evidence are authored from the normalized corpus without observing retrieval ranks or benchmark scores. Sources are the frozen CPython 3.12.10 documentation snapshot recorded in the corpus manifest.

## Questions and split

- Target: 30 answerable English questions, 20 dev and 10 test.
- Six questions carry explicit input scope: four dev and two test. Filters use verified metadata and are stated in the question/context before retrieval. Gold IDs/pages/answers must never become retrieval inputs.
- Direct, paraphrased and conditional questions are included; there is no mandated type ratio.
- Split is fixed before model retrieval. Questions in both splits can concern the same documentation family but do not repeat the same target fact. This is a small within-domain holdout, not unseen-document generalization.
- Authoring and automated evidence validation are AI-assisted. Human verification is not claimed without the student's actual review.
- Every question records the expected answer, a contiguous evidence quote and valid source/unit. Alternatives are recorded where another included passage supports the same answer. Validation checks occurrence after normalization.
- Test questions are prepared/validated before experiments, but their retrieval results must not be used to adjust configurations, filters, evidence or ranking.

## Controlled experiment

A uses fixed-size chunks; B uses paragraph-aware chunks; C reuses B's exact chunks, vectors and metadata and applies the supplied filter only when one exists. All strategies share corpus, embedding model/revision, query/document preprocessing, vector store, cosine metric and top-k=3.

A–B changes the whole declared chunking policy, including overlap differences; it does not isolate headings alone. B–C isolates metadata filtering. Without a query filter, B and C should retrieve equivalently under the same tie ordering.

## Selection rule and test gate

Choose on dev only, in order:

1. Highest Evidence Hit@3.
2. Highest Hit@1.
3. Lower measured p95 retrieval latency (record timing scope, cache and measurement noise).
4. For an exact remaining tie, prefer B when equivalent indexing/results do not require filters; otherwise use fixed order B, A, C.

Save finalized configurations, corpus/question hashes, the chosen strategy and reasons before test. Run test for all three finalized configurations while keeping the dev-selected choice fixed. Test grounding labels are never a selection criterion. If test results are used to fix/tune, disclose contamination and use a fresh holdout when possible.

## Retrieval metrics and traces

A hit requires a result from a valid gold source/unit and sufficient evidence in that single chunk. Correct source alone is insufficient. Hit@1 is the fraction of questions whose first result hits. Evidence Hit@3 is the fraction with at least one hit among the first three results. Report n/N and rates by split and strategy; failed queries stay in the denominator with their error logged.

Whitespace-normalized exact quote matching is an automatic check, not complete semantic adjudication. Accept documented alternative evidence. Equivalent evidence outside exact matching needs a separately documented reviewer decision; explain any adjusted score. Multi-chunk evidence union is supplementary and cannot replace the required single-chunk metric.

Save query text/ID/split, supplied/applied filter, configuration, retrieved chunk IDs/text/scores/metadata and hit decisions. Empty results are retrieval failures, not successful filter evidence. Do not inflate compliance by treating an empty result set as successful retrieval.

## Preselected coherence sample

Select ten source positions before chunk results are inspected, stratified deterministically across document IDs. For each document, use the earliest units in sorted order and choose the start of the middle eligible prose paragraph (at least 80 characters, excluding prompt examples, index fields and table/header markers; exact rule in build_benchmark.py). Add units in sorted round-robin order until ten positions are selected. `coherence_positions.json` is the authoritative list.

Inspect the chunk covering each anchor. If several cover it, choose the greatest start offset not exceeding the anchor, then chunk ID for a deterministic tie; do not choose the best-looking chunk. Score 0 (not understandable/cut meaning), 1 (partial/missing context), or 2 (self-contained and complete), with a reason. B and C share chunks, so their coherence judgments are explicitly reused observations.

## Metadata and grounding audits

Audit all ten test questions per strategy:

- Metadata: each returned top-three record, up to 90 records total. Check source, page/unit and relevant scope against normalized source/manifest; report complete-and-correct n/N, actual N and errors. Report filtered-query compliance with actual result counts; N/A if no filter query applies.
- Grounding: 30 query-strategy judgments of top-three sets: 0 no evidence, 1 incomplete conditions/context, 2 enough evidence to answer and trace source. Report counts and mean /2 with reasons. This does not measure LLM-generated answer accuracy.
- Labels declare reviewer/method and status. AI-proposed labels are not human-reviewed until the student checks them. Auto Hit@3 cannot substitute for these audits.

Analyze A–B and B–C alongside chunk count, lengths, ingestion/query time, cache and service cost. Explain two real failures with question/source IDs when present. If none occur, say so and use separate diagnostic cases without changing the holdout. Do not claim the winner is generally best beyond this corpus/benchmark.

