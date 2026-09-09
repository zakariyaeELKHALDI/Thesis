# Embedding and FAISS Index Decision and Schema

## Decision status

- **Status:** Accepted for production implementation; production outputs not yet generated
- **Decision date:** 2026-09-09
- **Embedding-index config schema version:** `1.0`
- **Index-mapping schema version:** `1.0`
- **Embedding-index summary schema version:** `1.0`
- **Configuration:** `configs/embedding-index-config.json`
- **Input contract:** `docs/corpus-chunking-decision-and-schema.md`
- **Audit evidence:** `notebooks/02_embedding_and_faiss_exploration.ipynb`

## Purpose

This document freezes the boundary between the validated retrieval chunks and later query retrieval. It records the evidence used to select the OpenAI embedding model, request behaviour, vector representation, normalisation rule, FAISS index and persistence contract.

Only `data/processed/corpus/chunks.jsonl` with its frozen SHA-256 fingerprint may enter corpus embedding and index construction. Evaluation questions may later be embedded as runtime queries, but they must never be added to the corpus index. Ground-truth answers must remain unavailable to embedding, indexing, retrieval and generation.

Retrieval depth `k` is deliberately not frozen here. It remains unresolved until a separate retrieval evaluation can compare candidate depths against the benchmark without using ground-truth answers during retrieval.

## Relationship to the reported baseline

Tophel et al. (2025) report OpenAI embeddings and FAISS similarity search. They do not report the exact embedding model, dimensions, wrapper settings, vector dtype, normalisation rule, FAISS index class, similarity metric, request batch size or retrieval depth.

The reconstructed baseline therefore preserves the reported components while classifying each missing value transparently:

| Setting | Frozen value | Classification |
|---|---|---|
| Embedding provider | OpenAI | Directly reported |
| Embedding integration | `langchain_openai.OpenAIEmbeddings` | Reconstructed LangChain implementation decision |
| Requested embedding model | `text-embedding-ada-002` | Reconstructed from the installed wrapper default |
| Dimensions | `1536` | Verified model output |
| `dimensions` argument | Omitted / `None` | Required compatibility choice for `text-embedding-ada-002` |
| Token encoding | `cl100k_base` | Verified locally and against API usage |
| Request batch size | `256` | Reconstructed operational decision |
| Vector dtype | contiguous NumPy `float32` | FAISS compatibility decision |
| Vector normalisation | Explicit L2 for corpus and query vectors | Reconstructed similarity decision |
| FAISS index | `IndexFlatIP` | Reconstructed exact-search decision |
| Score interpretation | Cosine similarity between unit vectors | Consequence of L2 normalisation plus inner product |
| Retrieval depth `k` | Not yet selected | Deferred to retrieval evaluation |

## Embedding-model decision

The paper identifies only "OpenAI embeddings". It does not provide a model name or a historical environment lock. The installed `langchain-openai 1.4.2` package exposes `text-embedding-ada-002` as the default model for `OpenAIEmbeddings`. This makes `text-embedding-ada-002` the closest observable LangChain baseline assumption in the verified environment.

`text-embedding-3-small` was considered because it is newer and cheaper, but selecting it would silently modernise an unspecified baseline component. It is therefore not used for the reconstructed baseline. A later embedding-model comparison may test it only as a separate extension with its own configuration and rebuilt index.

The selected model is an inference, not a claim about the exact unpublished model used by Tophel et al. This limitation must remain visible in the dissertation.

## Local token and request audit

The audit loaded the exact ordered 5,113-chunk export with SHA-256 `bce9702308a65b9e9ab3bcdb0781ad308d509b56904a95ec7bab4feffebca645`.

| Measure | Audited result |
|---|---:|
| Chunk records | 5,113 |
| Total chunk characters, including overlap | 848,280 |
| Total `cl100k_base` tokens | 303,045 |
| Minimum tokens per chunk | 2 |
| Median tokens per chunk | 48 |
| P95 tokens per chunk | 135 |
| Maximum tokens per chunk | 183 |
| Inputs above 8,192 tokens | 0 |
| Duplicate text occurrences | 5 |

The five duplicate text occurrences are retained because different chunks may contain the same text while preserving distinct source identifiers and page provenance. They must not be deduplicated before indexing.

The OpenAI embeddings endpoint documentation checked on 2026-09-09 permits at most 2,048 array inputs, 8,192 tokens per input and 300,000 combined tokens per request. These are external API constraints rather than properties of the research dataset. Account rate limits remain runtime-specific and are not frozen as experimental parameters.

## Batch-size decision

Three candidate batch sizes were measured against the exact ordered corpus:

| Batch size | Requests | Maximum inputs | Maximum tokens | Endpoint limits passed |
|---:|---:|---:|---:|---|
| 1,000 | 6 | 1,000 | 70,400 | Yes |
| 512 | 10 | 512 | 41,195 | Yes |
| 256 | 20 | 256 | 22,552 | Yes |

Batch size `256` is selected because it remains well below both endpoint limits, lowers the amount of work lost after a failed request and supports resumable staging. It requires more requests than the larger candidates, but the complete corpus is small enough that this is an acceptable reliability trade-off.

The production implementation must write and validate each completed batch before requesting the next one. It must honour server retry information, use the frozen retry settings and allow an interrupted run to resume only when the configuration fingerprint, input fingerprint, batch boundaries, model and completed batch fingerprints still match.

Pricing is not stored as a permanent production constraint because it can change externally. At the price observed during the audit, `$0.10` per million input tokens, a one-pass corpus embedding was estimated at `$0.030305` before retries. Current pricing should be checked again when costs are reported in the final dissertation.

## API and LangChain smoke-test evidence

A guarded paid smoke test embedded the first, middle and last chunk through the direct OpenAI API. It produced:

- requested model `text-embedding-ada-002`;
- observed response model `text-embedding-ada-002-v2`;
- response indices `[0, 1, 2]` in input order;
- matrix shape `(3, 1536)`;
- NumPy dtype `float32`;
- finite values for every vector component;
- raw and explicitly normalised vector norms equal to `1.0`;
- 195 locally predicted tokens and 195 API prompt tokens;
- and raw matrix SHA-256 `22a63cd752838295022a356a3b423dbe3cfe4aee79f8f60ccc42cd33d44d5bcf`.

The same three chunks were then embedded with the selected LangChain wrapper. The direct and wrapper matrices were exactly equal. Their paired cosine similarities were all `1.0`, their maximum absolute difference was `0.0` and their SHA-256 fingerprints matched.

This confirms that the wrapper preserves the tested input order and output values under the verified environment. It does not prove that the remote alias will always produce identical bytes in future runs.

## Vector representation and similarity

FAISS expects `float32` arrays. Every corpus and query matrix must therefore be converted to a finite, C-contiguous NumPy `float32` array before indexing or search.

The smoke-test vectors already had unit length, but this observed behaviour is not treated as an undocumented permanent provider guarantee. The implementation will explicitly apply `faiss.normalize_L2` to corpus and query vectors. Every resulting norm must be within `0.000001` of `1.0`.

`IndexFlatIP` performs exact inner-product search and does not require training. For L2-normalised vectors, inner product is equal to cosine similarity. This gives the score a clear interpretation while avoiding an approximate-index parameter that is unnecessary for only 5,113 vectors.

A synthetic local audit confirmed that `IndexFlatIP` and `IndexFlatL2` produced identical rankings for normalised vectors. It also confirmed that FAISS positions and scores remained exactly equal after writing and reloading the index.

## Production outputs

The embedding and index stage will publish four final files:

| Output | Purpose |
|---|---|
| `data/processed/embeddings/chunk-embeddings.npy` | Ordered, finite, L2-normalised `(5113, 1536)` contiguous `float32` matrix |
| `data/processed/index/chunks.faiss` | Persisted `IndexFlatIP` containing one vector per ordered chunk |
| `data/processed/index/chunk-mapping.jsonl` | Explicit mapping from every FAISS position to the corresponding chunk identifiers and provenance keys |
| `data/processed/audit/embedding-index-summary.json` | Input, configuration, request, batch and output fingerprints plus structural audit totals |

All four outputs and resumable staging files contain derived information from a copyrighted source or externally generated vectors and must remain excluded from Git. The summary must be published last. Existing final outputs must not be replaced without an explicit overwrite option.

The embedding matrix and index must describe the same ordered vectors. Reloading the `.npy` matrix and FAISS index must reproduce the accepted dimensions, vector count and search behaviour before publication succeeds.

## Index-mapping record contract

Every line of `chunk-mapping.jsonl` must contain one JSON object with these fields:

| Field | Type | Meaning |
|---|---|---|
| `index_mapping_schema_version` | string | Fixed value `1.0` |
| `index_position` | integer | Zero-based FAISS position and embedding-matrix row |
| `chunk_id` | string | Stable identifier of the corresponding retrieval chunk |
| `parent_record_id` | string | Stable identifier of the parent retrieval-region record |
| `source_id` | string | Validated source identifier |
| `pdf_page_index` | integer | Zero-based source PDF page index |
| `pdf_page_number` | integer | One-based source PDF page number |
| `printed_page_number` | integer or null | Printed page number when available |

Mapping records must follow exact contiguous position order from `0` through `5112`. Their chunk identifiers must match `chunks.jsonl` in the same order. The mapping must not duplicate chunk text because the validated chunk export remains the canonical retrieval-text source.

## Required validation failures

Production embedding and index construction must stop without publishing final outputs if:

- the input path escapes the authorised `data/processed/corpus/` boundary;
- the chunk export or chunking-summary fingerprint differs;
- record count, schema, ordering, identifiers or expected token audit differs;
- any input text is empty or any chunk identifier is duplicated;
- a batch boundary, model or configuration differs from resumable staging evidence;
- the API returns missing, duplicated or out-of-order response indices;
- an embedding has the wrong dimension or contains a non-finite value;
- conversion does not produce a C-contiguous `float32` matrix;
- any vector remains outside the accepted norm tolerance after explicit L2 normalisation;
- embedding rows, FAISS positions and mapping records do not remain one-to-one;
- the FAISS index type, dimension or vector count differs after reload;
- an index search changes positions or scores after persistence and reload;
- a final output already exists and explicit overwrite was not requested;
- serialised JSON contains a non-finite number;
- or the summary would be published before every other output has been validated.

A failed API request may leave validated staging files for safe resumption, but it must not publish a partial embedding matrix, index, mapping or final summary.

## Deterministic local serialisation

Local metadata must use UTF-8, sorted JSON keys, `ensure_ascii=False`, `allow_nan=False` and a final newline. JSONL must contain one compact object per line. NumPy and FAISS files must be written through temporary paths and reloaded for validation before publication.

The mapping and summary are deterministic for a fixed input, configuration and embedding matrix. The externally generated embedding bytes are observed results rather than guaranteed deterministic rebuild targets.

## Reproducibility boundary

The smoke-test fingerprint records the API response observed during this audit. It is not treated as a permanent regeneration requirement because `text-embedding-ada-002` is a remotely served model alias. Production exports must record the requested model, observed dimensions, exact input fingerprint, configuration fingerprint and every generated artifact fingerprint.

A later regeneration may be structurally valid while producing a different embedding fingerprint. Such a change must be recorded as a new external-model observation and must rebuild the complete FAISS index. Embeddings from different runs must never be mixed within one index.

## Retrieval depth remains unresolved

The paper mentions `search_kwargs` but does not report the value of `k`. Selecting `k` from convenience or a library default would hide another reconstruction assumption.

The production index can be built without selecting `k`. A separate retrieval audit must later compare justified candidate depths while keeping the corpus, chunks, embedding model, vector normalisation and FAISS index fixed. The selected depth must then be recorded in a retrieval configuration before benchmark generation begins.

## Assessment alignment

This decision record supports the dissertation Approach section by separating directly reported methodology from reconstructed choices, naming package versions and model settings, documenting alternatives, and defining reproducible input and output contracts. It also supports the Evaluation and Limitations sections by recording the remote-model reproducibility boundary and preventing `k` from being selected without evidence.

## Relationship to the research questions

| Research question | Effect of this decision |
|---|---|
| RQ1: Apply the methodology of Tophel et al. | Preserves the reported OpenAI and FAISS components while exposing the unreported model and similarity assumptions |
| RQ2: Compare newer LLM versions | Freezes the corpus embeddings and index so generator-model comparisons do not change retrieval representation |
| RQ3: Test controlled parameter changes | Keeps retrieval depth unresolved for a documented one-variable-at-a-time evaluation |
| RQ4: Address baseline limitations | Allows a later embedding or formula-aware extension to use a separate configuration without being presented as baseline replication |

## Later experimental use

The selected embedding model, vector representation and FAISS index form the reconstructed baseline for RQ1. They must remain fixed throughout the main generator-model comparison for RQ2. Any later embedding-model comparison must use a separate configuration identifier, regenerate every corpus vector and rebuild its own index.

Official API evidence checked for this decision:

- [OpenAI embeddings API reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create)
- [OpenAI `text-embedding-ada-002` model page](https://developers.openai.com/api/docs/models/text-embedding-ada-002)
