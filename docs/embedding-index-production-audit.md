# Production Embedding and FAISS Index Audit

## Status

- **Audit status:** Passed
- **Production and audit date:** 2026-09-12
- **Implementation commit:** `b2c11d1`
- **Frozen configuration:** `configs/embedding-index-config.json`
- **Configuration SHA-256:** `7a36bb84bfb190bed02767a30a000e84eb2d512b80a5af2e2832b25849d3bbf2`
- **Machine-readable evidence:** `results/metrics/embedding-index-production-audit.json`
- **Retrieval depth `k`:** Still unresolved pending retrieval evaluation

## Purpose

This record documents the first complete production embedding and FAISS index build. It separates the production run from the later independent checks so that successful output from the generating module is not treated as sufficient evidence on its own.

The frozen configuration is not edited after this run. Its status field records the stage at which the decision was frozen, and its SHA-256 fingerprint is embedded in every staging metadata record and in the final summary. Changing it only to update wording would break that traceability.

## Fixed production input

| Item | Verified value |
|---|---|
| Retrieval chunks | 5,113 |
| Input tokens | 303,045 using `cl100k_base` |
| Chunk export SHA-256 | `bce9702308a65b9e9ab3bcdb0781ad308d509b56904a95ec7bab4feffebca645` |
| Chunking summary SHA-256 | `d31c5ed3a3adca6d4f4f839e404c6582ef4419b385ffc5e5b49d253ac21f96d5` |
| Requested embedding model | `text-embedding-ada-002` |
| Embedding dimensions | 1,536 |
| Batch size | 256 |
| Completed batches | 20 |
| Vector format | C-contiguous NumPy `float32` |
| Normalisation | Explicit L2 normalisation |
| FAISS index | Exact `IndexFlatIP` |

## Published production artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `data/processed/embeddings/chunk-embeddings.npy` | 31,414,400 | `5691dd58b3de2c79985ebd403f80e0c70a7d95cb2609c835325062f7bad2c1c0` |
| `data/processed/index/chunks.faiss` | 31,414,317 | `e049fa90324346c764c849071dbc49b7bc0cce125bc2fe8331a6f560fea8bdd1` |
| `data/processed/index/chunk-mapping.jsonl` | 1,495,044 | `fc0515de3e983988f04fe1e5d7ee80cfc85eae3cce8e3153555a02e04b274a4f` |
| `data/processed/audit/embedding-index-summary.json` | 1,963 | `c9e5118ee9ecaf15339883e09becd0a046b5019287a790264be5a3dabe9a64eb` |

These generated files remain excluded from Git because they contain derived textbook information or externally generated vector values. Their exact fingerprints and structural results are retained in this document and in the versioned metrics record.

## Initial production run

The implementation at commit `b2c11d1` embedded all 5,113 chunks in 20 sequential batches. Every completed raw batch was written as a NumPy matrix plus a metadata record before the next batch started. The final normalised matrix, FAISS index, mapping and summary were published only after every batch had completed.

The initial run reported no incomplete batch, retry failure, dimension error, non-finite vector or publication error.

## Independent audit

A separate read-only audit script was run after publication. The script did not import `geotech_rag.embedding_index`, did not access `.env`, did not make an API request and did not modify the generated files.

The production LangChain wrapper returns an ordered vector list rather than exposing the raw API response indices. Production therefore validates one output vector per ordered input and relies on the wrapper ordering contract. This assumption is supported by the earlier smoke test, where the direct API and wrapper matrices were exactly equal for the same first, middle and last chunks. Raw response-index validation applies only to that direct API smoke test.

It independently confirmed:

- all fixed input, configuration and output fingerprints;
- all four final file sizes;
- exactly 20 staged matrices and 20 matching metadata records;
- continuous batch coverage from position `0` through `5112`;
- exact batch chunk-ID, text, token-count and matrix fingerprints;
- exact reconstruction of the final matrix from the staged raw vectors followed by L2 normalisation;
- matrix shape `(5113, 1536)`, dtype `float32`, C-contiguous storage and finite values;
- vector norms between `0.9999997019767761` and `1.0000003576278687`;
- exact mapping order and provenance for all 5,113 positions;
- FAISS class `IndexFlatIP`, dimension 1,536 and total vector count 5,113;
- exact reconstruction of every indexed FAISS vector against its matrix row; and
- five representative FAISS searches consistent with exhaustive NumPy inner-product search.

## Overwrite and deterministic-regeneration checks

Repeating the production command without `--overwrite` was rejected before any output was changed. This confirms that existing production artifacts cannot be replaced accidentally.

The command was then repeated with `--overwrite` while the process-level OpenAI credential was replaced by an invalid placeholder. All 20 batches were reported as `resumed` and zero batches were reported as `embedded`. The command therefore could not depend on a new paid API request.

The rebuilt matrix, FAISS index, mapping and summary reproduced all four original SHA-256 fingerprints exactly. This verifies deterministic local reconstruction from the retained staging evidence.

## Staging-retention decision

The 40 staging files are retained. They contain 20 raw matrix files and 20 metadata files. Their combined metadata projection SHA-256 is `0f7648b63fb359bea7ff52d3a199d4bb3069722bc39d32b17551abbd3466a64d`.

Retention is justified because the files allow interrupted work to resume and allow the final normalised matrix and index to be recreated without another external embedding request. The storage cost is small relative to the benefit for reproducibility. Staging must not be mixed with another configuration, chunk fingerprint or embedding run.

## Reproducibility boundary

The local regeneration is deterministic when it uses these exact staged vectors. A completely new call to the remotely served model alias may produce different bytes in the future, even with the same text and requested model. Such a run must be recorded as a new external-model observation and must rebuild every vector and the complete FAISS index.

## Retrieval boundary

This audit validates vector creation and exact similarity-index construction. It does not select or validate retrieval depth `k`, question embedding, ranking usefulness or answer generation. Those decisions belong to the next retrieval evaluation so they are not selected from convenience or from one undocumented library default.

## Assessment alignment

This evidence supports the dissertation Approach section by connecting the frozen method to a specific tested Git commit and exact input. It supports implementation quality through validation, failure handling, resumability and independent checking. It also supports the Evaluation and Limitations sections by separating deterministic local reconstruction from the uncertainty of a remotely served embedding model.
