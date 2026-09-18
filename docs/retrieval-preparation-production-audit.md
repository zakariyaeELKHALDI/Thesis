# Retrieval-preparation production audit

Audit date: 2026-09-18

Configuration identifier: `benchmark-retrieval-depth-evaluation-v1`

## Purpose

This document records the production preparation of the blinded
retrieval-depth evaluation. The retrieval protocol and candidate depths
were frozen before the evaluation questions were embedded or searched.
The frozen configuration was not changed during production.

This checkpoint covers question embedding, exact FAISS search at the
judgement-pool depth, creation of the rank-blinded judgement template,
and independent verification before any relevance grade was entered.

## Temporary API incident

The first production embedding request returned HTTP 404 before any
output artifact was created. A read-only routing diagnostic confirmed
the standard OpenAI API endpoint and no alternative base URL, Azure
configuration or proxy.

The configured model was then successfully retrieved and listed for the
same API key. A seven-token direct embedding probe returned one finite
1,536-dimensional vector from `text-embedding-ada-002-v2`. The exact
production LangChain wrapper also returned one finite 1,536-dimensional
vector. A controlled production retry then passed without any
configuration change.

The incident is therefore recorded as a non-persistent HTTP 404 resolved
by controlled retry. It did not cause a model substitution, protocol
change or partial production output.

## Production result

| Check | Result |
|---|---:|
| Evaluation questions | 20 |
| Query embedding tokens | 1675 |
| Query matrix shape | 20 x 1536 |
| Query matrix dtype | `float32` |
| Minimum query norm | 0.9999998211860657 |
| Maximum query norm | 1.000000238418579 |
| FAISS index class | `IndexFlatIP` |
| Corpus vectors | 5113 |
| Judgement pool depth | 20 |
| Ranked candidates | 400 |
| Blinded judgement records | 400 |
| Completed judgements | 0 |
| Selected retrieval depth | unresolved |

## Production artifact fingerprints

| Artifact | SHA-256 |
|---|---|
| Query embedding matrix | `fddd63335e250d875781890cec48cc646165e221d9b0cabdd22f489dc35be882` |
| Ranked candidate pool | `1713a14a25433abde59791eede325f33605da86da85019e2cda3597ec0ffc114` |
| Initial blinded judgement template | `bbc72d5e423bf5d745ec93cb9b6de952520135039a3f6dac5132b3a784b718e1` |
| Retrieval-preparation summary | `6083b0048d45905cd6724b472a6ec6f577a16f5c98984e9bde4fa8b01402e040` |

The generated matrix, candidate pool, judgement file and summary remain
ignored by Git. They contain either derived evaluation data or private
copyrighted question and chunk text. Only this text-free audit evidence
is tracked.

## Independent verification

The independent audit reproduced all 20 searches directly
against the production `IndexFlatIP` index. Every saved index position
and similarity score matched the fresh FAISS result. It also verified:

- all 12 frozen input and production fingerprints;
- the `(20, 1536)` float32, C-contiguous query matrix;
- finite values and the configured L2-normalisation tolerance;
- 400 unique question-chunk pairs and judgement identifiers;
- deterministic rank order in the candidate pool;
- deterministic hash order in the blinded judgement template;
- equality of candidate and judgement identifier sets;
- zero completed grades and zero judgement notes;
- absence of rank, similarity and provenance fields from the judgement
  file;
- absence of scored metrics before annotation.

No API request was made during the independent audit.

## Leakage and decision boundary

Ground-truth answers were not accessed. No answer was generated, and no
evaluation question was added to the corpus index. The assessor-facing
file contains only the question, retrieved corpus chunk, blinded
identifier and empty judgement fields.

Retrieval depth `k` remains unresolved. It may be selected only after all
400 blinded relevance judgements are completed and the frozen scoring
rule is applied.

The machine-readable companion record is
`results/metrics/retrieval-preparation-production-audit.json`.
