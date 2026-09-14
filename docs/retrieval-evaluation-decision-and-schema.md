# Retrieval-depth evaluation decision and schema

## Decision status

This document freezes the retrieval-depth evaluation protocol before any benchmark question is embedded or submitted to FAISS. The selected retrieval depth remains unresolved until the blinded relevance judgements and predefined metrics have been completed.

Configuration identifier: `benchmark-retrieval-depth-evaluation-v1`
Configuration SHA-256: `c99ee459c389407dd77e710b8b2ec5081d51889d0d1e5a713d5dcdf9d5d68c0f`

At the point of this decision:

- benchmark question embeddings generated: no;
- FAISS benchmark search run: no;
- relevance judgements created: no;
- solution-manual ground truth accessed: no.

This timing matters because candidate depths, metrics and the selection rule must not be changed after seeing which setting performs best.

## Frozen inputs

The evaluation uses the existing 20-question benchmark and the independently audited production index.

| Input | Frozen evidence |
|---|---|
| Evaluation-question configuration | `bae9572fdc3007b857e7d8af3c2e01a0ef0f5b9b8767f187c8de5b31771007c6` |
| Local question file | `ec7c151ba709eb6e55d37290dc8fb349a5def8b204f439b8e354f71d0b97d599` |
| Question projection | `d785ffa3164d330f689512e4b52794a6316fa94596287bbb60ddf1378e27b343` |
| Question count | `20` |
| Embedding-index configuration | `7a36bb84bfb190bed02767a30a000e84eb2d512b80a5af2e2832b25849d3bbf2` |
| Production index summary | `c9e5118ee9ecaf15339883e09becd0a046b5019287a790264be5a3dabe9a64eb` |
| Corpus chunks | `bce9702308a65b9e9ab3bcdb0781ad308d509b56904a95ec7bab4feffebca645` |
| FAISS index | `e049fa90324346c764c849071dbc49b7bc0cce125bc2fe8331a6f560fea8bdd1` |
| Index mapping | `fc0515de3e983988f04fe1e5d7ee80cfc85eae3cce8e3153555a02e04b274a4f` |

Evaluation questions remain runtime queries. They are not added to the 5,113-chunk corpus index.

## Why retrieval depth must be evaluated

The replication paper mentions `search_kwargs` but does not report the retrieval depth `k`. The installed LangChain interface has `k=4` as its default, so 4 is retained as the reference baseline. Treating that default as automatically optimal would still leave an untested reconstruction assumption.

The candidate depths are `[1, 2, 3, 4, 5, 8, 10]`. Values 1 through 5 provide a detailed comparison around the LangChain reference depth. Values 8 and 10 test whether additional context improves evidence coverage before context size becomes unnecessarily large.

The exact installed reference is:

```text
langchain-core 1.5.3
VectorStore.similarity_search(self, query: 'str', k: 'int' = 4, **kwargs: 'Any') -> 'list[Document]'
```

## Query embedding and search contract

All 20 ordered question texts will be embedded in one `embed_documents` request using the same `text-embedding-ada-002` model and 1536-dimensional representation used for the corpus.

The returned matrix must:

- contain one vector per ordered question;
- have shape `(20, 1536)`;
- use `float32` values in C-contiguous memory;
- contain only finite values;
- receive explicit L2 normalisation using the frozen norm tolerance;
- remain separate from the corpus embedding matrix and FAISS index.

Each normalised query is searched once against the existing `IndexFlatIP` index at pool depth 20. Inner product is interpreted as cosine similarity because both query and corpus vectors are L2-normalised. Results must be in descending score order and joined to provenance through the mapping field `index_position`.

Candidate metrics are calculated from nested prefixes at `k = 1, 2, 3, 4, 5, 8, 10`. This avoids repeated searches producing different candidate sets.

## Blinded relevance judgements

The judgement unit is one question and one retrieved corpus chunk. Only the question text and chunk text may be shown during judgement. Retrieval rank, similarity score and candidate-depth membership must be hidden.

Pairs are placed in deterministic blinded order using SHA-256 over the configuration identifier, question identifier and chunk identifier. This preserves reproducibility without allowing rank to influence the judgement.

The grades are:

| Grade | Label | Meaning |
|---:|---|---|
| 0 | Not relevant | Does not help answer the question or only shares unrelated terminology |
| 1 | Supporting evidence | Provides a useful definition, assumption, relationship or background step |
| 2 | Direct evidence | Provides the central equation, method, worked procedure or concept needed to answer the question |

A short judgement note is required for grades 1 and 2. Solution-manual answers and model-generated answers must remain unavailable throughout this process.

## Metrics

The primary metric is direct-evidence hit rate at each candidate depth. A hit requires at least one grade-2 chunk in the first `k` results.

The evaluation also reports:

- useful-evidence hit rate, using grades 1 or 2;
- Wilson 95% intervals for both hit rates;
- reciprocal rank of the first direct-evidence chunk;
- graded nDCG using gain `2^grade - 1`;
- useful-evidence precision, using grades 1 or 2;
- mean retrieved context tokens and characters.

Conventional recall is not reported because the complete 5,113-chunk corpus is not exhaustively judged. Calling the pooled judgements complete corpus relevance would overstate the evidence.

## Candidate-range adequacy gate

The top-20 pool is larger than the largest candidate depth so it can test whether the proposed range ends too early.

If a question has no direct evidence within ranks 1 to 10 but gains direct evidence within ranks 11 to 20, no `k` will be selected. The protocol must first be versioned with expanded candidate depths. Pool depth 20 is therefore diagnostic and is not itself a candidate in this version.

## Frozen selection rule

If the candidate-range adequacy gate passes:

1. retain the depths with the maximum direct-evidence hit count;
2. among those, retain the depths with the maximum useful-evidence hit count;
3. select the smallest remaining depth.

This gives priority to retrieving the evidence needed for an answer, then avoids adding context when it does not improve question coverage. MRR, nDCG and precision remain interpretation metrics rather than post-result tie-breakers.

## Leakage and storage controls

- Evaluation questions may be embedded only as runtime queries.
- Neither question text nor question vectors may be added to the corpus index.
- Ground-truth answers must remain unavailable during retrieval and relevance judgement.
- No answer generation is allowed during this evaluation stage.
- Local files containing benchmark question or textbook chunk text remain ignored by Git.
- Tracked metric files must contain identifiers, counts and measurements only, not copyrighted question or chunk text.

The blinded judgement file is local evidence and remains under `data/raw/evaluation/`. Query vectors, ranked candidates and the run summary remain under ignored `data/processed/` paths. Aggregate metrics, a metrics table and the completed exploration notebook may later be tracked after their contents have been checked for text leakage.

## Reproducibility boundary

The query embedding call uses a remotely served model alias. The query matrix fingerprint records the observed response but cannot be treated as a permanent byte-for-byte regeneration requirement. Reproduction instead requires the requested model, input fingerprints, dimensions, ordering, normalisation rules, index fingerprints and generated output fingerprints to be recorded.

The 20 benchmark questions are used to calibrate `k`, so the selected depth is an internal benchmark calibration rather than an estimate from an independent holdout set. This limitation must be stated when retrieval and answer-generation results are interpreted.
