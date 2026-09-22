# Retrieval-Depth Selection Decision

## Status

This decision was frozen on 22 September 2026 after completion of the
version 3 retrieval-depth evaluation and before implementation of the
production retriever or generation of benchmark answers.

The evidence-producing result is fixed by Git commit:

`1aa3f3a47cbf873864ff1d0021743e8dccf2d770`

This document records a post-evaluation production decision. It does not
present the selected depth as having been known before the retrieval
evaluation.

## Decision

The reconstructed baseline RAG pipeline will use a retrieval depth of:

`k = 30`

For each query, the production retriever must return the thirty
highest-ranked corpus chunks from the frozen exact FAISS index.

The retrieved records must retain their original rank order, similarity
scores, chunk identifiers and provenance. Retrieval-specific metadata may
be retained for reproducibility, but it must not be represented as source
content in the generated answer.

No unrecorded reranking, filtering, deduplication, compression or
rank-based truncation may reduce the effective retrieval depth.

If a later context-management step removes or transforms any retrieved
record, that step must use a separate versioned configuration and must not
be described as the evaluated baseline retrieval procedure.

## Selection rule

The version 3 evaluation applied the selection rule frozen before its
expanded search:

1. retain candidate depths with the maximum number of questions containing
   direct evidence;
2. among any tied depths, retain those with the maximum number containing
   useful evidence;
3. select the smallest remaining depth.

MRR, graded nDCG, useful-evidence precision and context size were reported
as diagnostic measures and were not used as selection tie-breakers.

At depth 25, direct evidence was retrieved for 18 of the 20 evaluation
questions. At depth 30, direct evidence was retrieved for 19 questions.
Depth 30 therefore uniquely maximised the primary selection criterion.

Useful evidence was retrieved for all 20 questions at depth 30.

## Candidate-range adequacy

The version 3 adequacy gate tested whether a question without direct
evidence in ranks 1-30 gained direct evidence in the diagnostic tail at
ranks 31-40.

The gate did not trigger.

Direct evidence was present for 19 questions through rank 30 and remained
present for 19 questions through rank 40. No question first gained direct
evidence in the diagnostic tail.

The candidate range is therefore adequate under the frozen protocol, and
no further retrieval-depth expansion is required for the reconstructed
baseline.

Question identifier `6.6b` had no direct evidence through rank 40. This is
retained as a retrieval or corpus-coverage limitation. It is not treated as
a reason to expand depth automatically because the diagnostic tail supplied
no evidence that a deeper search would resolve the case.

## Selected-depth evidence

At `k = 30`, the aggregated retrieval results are:

| Measure | Result |
|---|---:|
| Direct-evidence hit count | 19 / 20 |
| Direct-evidence hit rate | 0.95 |
| Direct-evidence hit-rate Wilson 95% interval | 0.7639 to 0.9911 |
| Useful-evidence hit count | 20 / 20 |
| Useful-evidence hit rate | 1.00 |
| Useful-evidence hit-rate Wilson 95% interval | 0.8389 to 1.0000 |
| Direct-evidence MRR | 0.273125 |
| Graded nDCG | 0.659317 |
| Useful-evidence precision | 0.563333 |
| Mean retrieved context tokens | 1935.85 |
| Mean retrieved context characters | 5289.85 |

These results apply to the frozen 20-question benchmark, textbook corpus,
chunking configuration, embedding model, vector normalisation procedure and
FAISS index. They do not establish that depth 30 is universally optimal for
other corpora or retrieval systems.

## Context and precision trade-off

Depth 30 improves direct-evidence coverage relative to depth 25, but it also
retrieves more non-useful material.

Useful-evidence precision falls to approximately 0.563 at depth 30, while
the mean retrieved context reaches approximately 1,936 tokens. This creates
a cost and context-management trade-off.

The higher depth is accepted for the reconstructed baseline because the
frozen selection rule prioritises direct-evidence coverage and depth 30 is
the only evaluated candidate that reaches 19 direct-evidence hits.

The later prompt and generator configuration must be checked against this
retrieved-context size. A context-window or token-budget problem must not be
solved by silently reducing the selected depth. Any such change would
represent a new experimental configuration.

## Production retrieval invariants

The production retrieval configuration must preserve:

- retrieval depth `30`;
- the authorised processed corpus;
- the frozen 5,113-chunk collection;
- the frozen corpus embedding matrix;
- the existing exact FAISS index;
- inner-product similarity over L2-normalised vectors;
- the same query-embedding model and vector normalisation procedure;
- descending similarity-score order;
- zero-based index-position mapping and one-based displayed retrieval rank;
- complete chunk and source provenance;
- separation between retrieval content and ground-truth answers.

The production retriever must validate the configured artifact
fingerprints before answering benchmark questions.

## Frozen evidence lineage

The decision is supported by the following version 3 artifacts:

- configuration SHA-256:
  `5d847a027c098694d47cabc9dd2f73336a98c1d8b1c5e0c645d89e5c06dd84cb`;
- query-embedding matrix SHA-256:
  `fddd63335e250d875781890cec48cc646165e221d9b0cabdd22f489dc35be882`;
- FAISS index SHA-256:
  `e049fa90324346c764c849071dbc49b7bc0cce125bc2fe8331a6f560fea8bdd1`;
- ranked candidate pool SHA-256:
  `6045c15a6ac312deda123ea998a890a94c9c167595eba43b09f419ed35d46b80`;
- completed judgement SHA-256:
  `cbe0868da7c7637110de0f7aa57f3c4276939fde245fa1c2b8cf2556867366fe`;
- judgement-completion audit SHA-256:
  `5a825f9e30dad6501832507f9b8fa164a4bcbfa83083c8279093f9c6defc9d48`;
- metrics SHA-256:
  `4415459e535e832a84d87a468eb4ced21660be8489bfbb22248cf84f3ed3c2b4`;
- metrics-table SHA-256:
  `306baa7eb77ae06ad019a1f9976f1110944260d4cbae67dc6206c41cf7bb78cd`;
- independent scoring-audit SHA-256:
  `01d84622e2d4c505e96d4d625945f206779c4401f42a1ede89c9ab64fdd5c563`;
- final evaluation-notebook SHA-256:
  `d57c00e73e0b43fa425428421330866a51327f8f696682c4bc5206463678cb68`.

## Relationship to earlier records

Earlier architecture, embedding-index and evaluation configuration files
correctly state that retrieval depth was unresolved when those artifacts
were frozen.

Those historical records must remain unchanged. Their unresolved fields
show that `k` was not chosen before the evaluation evidence existed.

This decision record supersedes that unresolved status only for subsequent
production implementation and benchmark generation.

The version 1, version 2 and version 3 retrieval-evaluation configurations
also remain unchanged. Their null selected-depth fields describe the
protocol state frozen before scoring and must not be rewritten using the
observed result.

## Remaining implementation boundary

This decision freezes the retrieval depth but does not yet implement the
production RAG pipeline.

Before benchmark answer generation begins:

1. a separate production retrieval configuration must record `k = 30` and
   the required artifact fingerprints;
2. a reusable production retrieval module must validate and load the frozen
   index, mapping and embedding configuration;
3. automated tests must confirm deterministic ranking, record counts,
   provenance and refusal of incompatible artifacts;
4. the baseline prompt and insufficient-evidence behaviour must be frozen;
5. generator models, generation settings, retry rules and result schemas
   must be frozen;
6. all implementation and configuration changes must be committed before
   benchmark generation.

## Assessment alignment

This decision supports the dissertation by providing a transparent link
between the experimental evidence and the implemented system parameter.

It also records the main trade-off rather than presenting the selected
depth as unconditionally beneficial: depth 30 improves direct-evidence
coverage but increases context size and reduces evidence precision.

Keeping the pre-evaluation protocols unchanged while adding this
post-evaluation decision preserves chronological traceability and helps
separate planned methodology, observed results and implementation choices.
