# Corpus Chunking Decision and Record Schema

## Decision status

- **Status:** Implemented and verified for production use
- **Decision date:** 2026-09-03
- **Implementation verification date:** 2026-09-06
- **Chunking-config schema version:** `1.1`
- **Chunk-record schema version:** `1.0`
- **Chunking-summary schema version:** `1.0`
- **Configuration:** `configs/chunking-config.json`
- **Input contract:** `docs/corpus-record-and-export-schema.md`

## Purpose

This document freezes the boundary between the validated positioned-text export and later embedding and FAISS indexing. It distinguishes the values reported by Tophel et al. (2025) from the unreported values reconstructed for this thesis, records the evidence used to select those values and defines the provenance that every retrieval chunk must retain.

Only `data/interim/corpus/regions.jsonl` may enter this stage. The raw PDF, page-audit export, evaluation questions, ground-truth answers, excluded textbook problems, selected answers and Index must never become chunking input.

## Relationship to the reported baseline

Tophel et al. (2025) report `CharacterTextSplitter`, `chunk_size = 200` and `chunk_overlap = 10`. They do not report the separator, length function, whitespace behaviour, separator-retention behaviour, oversized-split policy or dependency version.

The reconstructed baseline therefore preserves the three reported values and explicitly freezes every missing value:

| Setting | Frozen value | Classification |
|---|---|---|
| Splitter class | `CharacterTextSplitter` | Directly reported |
| Chunk size | `200` | Directly reported |
| Chunk overlap | `10` | Directly reported |
| Separator | Single newline, `"\n"` | Reconstructed from the positioned-line record structure |
| Length function | Python `len` | Unreported installed default |
| Keep separator | `false` | Unreported installed default |
| Add start index | `true` | Provenance extension; does not alter chunk text |
| Strip whitespace | `false` | Preservation decision supported by the control-character audit |
| Protect embedded line newlines | `true`, using temporary `U+E000` | Provenance protection required by the positioned-line audit |
| Separator is regular expression | `false` | Unreported installed default |
| Package version | `langchain-text-splitters 1.1.2` | Locked project environment |
| Oversized-chunk policy | Stop with an error | Reconstructed validation rule |

No extra Unicode normalisation, formula reconstruction, dehyphenation or semantic rewriting is performed during chunking.

## Candidate audit

The audit used the exact frozen 670-record region export with SHA-256 `0bb7225c7f403d7c15fdaac2a58cc5a847d414dc48ce11668a2eaecde1cbb03a`. It compared three interpretations while holding the reported size and overlap values constant.

| Candidate | Separator | Chunks | Mean characters | P95 | Maximum | Chunks over 200 | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Installed default | `"\n\n"` | 670 | 1,258.730 | 2,512 | 3,779 | 653 | Rejected |
| Positioned-line boundary with default stripping | `"\n"` | 5,113 | 165.907 | 199 | 200 | 0 | Separator selected; stripping required a separate integrity check |
| Strict character boundary | `""` | 4,743 | 186.080 | 200 | 200 | 0 | Reserved for possible sensitivity analysis |

The installed double-newline default did not split any source record because the production extraction joins positioned lines with single newlines. It therefore failed to realise the reported 200-character intention on this corpus.

Strict character splitting enforced the numerical size and overlap more mechanically, but it could cut through words, formula sequences and table values. The single-newline candidate also produced no oversized chunks and best matched the positioned-line design. It was selected provisionally and then subjected to separate whitespace and embedded-control integrity audits before production use.

## Whitespace-preservation audit

The first separator comparison retained the installed `strip_whitespace=True` default. A focused follow-up audit showed that Python's broad `str.strip()` classification removed seven boundary characters from six chunks: four ordinary spaces, two `U+000B` controls and one `U+000C` control. The three controls belong to the mathematical-symbol evidence deliberately preserved by the extraction stage.

| Setting | Chunks | Total characters | Chunks containing formula controls | Formula controls removed | Projection SHA-256 |
|---|---:|---:|---:|---:|---|
| `strip_whitespace=True` | 5,113 | 848,283 | 2,118 | 3 | `77f85fc269745a0dd2eeab3404f2f03b128ba52f7c0e9978c2d1fa622234b784` |
| `strip_whitespace=False` | 5,113 | 848,290 | 2,119 | 0 | `b72dcd0ef4bc7c23edbcd976630b195bf926291debc3517f3596039f68f30021` |

Both settings produced a maximum chunk length of 200, no oversized chunks and chunk text found exactly in its parent record. Disabling stripping changes neither the number of chunks nor the reported size and overlap settings. It is therefore selected because it prevents the chunking stage from silently reversing the established text-preservation decision. This is recorded as a transparent reconstruction, since the paper does not report whitespace behaviour. The values in this table describe the provisional raw-newline candidate; the later positioned-line control audit changed five parent projections without reversing the whitespace decision.

## Embedded-control boundary audit

The first real-source integration run exposed a distinction that was not visible in the aggregate separator audit. Forty-three positioned-line records contain line-break control values inside their extracted line text: 33 contain `U+000A` and 10 contain `U+000D`. These values originate inside PyMuPDF line records and belong to the preserved mathematical-font evidence; they are not the structural newline inserted between two positioned lines.

Using raw `U+000A` as the splitter separator therefore gave one character two meanings. Although every emitted chunk remained a valid parent substring, the raw candidate began inside a positioned line three times, ended inside a positioned line three times and affected five parent records. This contradicted the frozen requirement that positioned lines remain atomic.

The production implementation protects only embedded `U+000A` values with one temporary private-use character, `U+E000`, before splitting. The source was checked to confirm that `U+E000` does not already occur. Because both values have length one, parent offsets remain unchanged. The original `U+000A` values are restored before validation, serialisation, embedding or any other downstream use. Embedded `U+000D` values remain unchanged because they are not the configured separator. This mechanism is structural protection rather than text normalisation or formula reconstruction.

| Candidate | Chunks | Total emitted characters | Starts inside lines | Ends inside lines | Parents with boundary problems | Projection SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| Raw single-newline splitting | 5,113 | 848,290 | 3 | 3 | 5 | `b72dcd0ef4bc7c23edbcd976630b195bf926291debc3517f3596039f68f30021` |
| Positioned-line-aware protected splitting | 5,113 | 848,280 | 0 | 0 | 0 | `88aca7a9a60c21713f3e6d732cb17f6380776d1c2f0c06157f3414c018f09c4b` |

The ten-character difference is caused by changed best-effort overlap formation around the five affected parents, not by deletion from the source records. The protected result retains all 39,934 positioned lines, preserves 2,119 chunks containing formula controls and emits no `U+E000` value. It is therefore the accepted production candidate.

## Overlap interpretation

The configured overlap remains `10`, as reported by the paper. `CharacterTextSplitter` applies this value as a target while merging separator-delimited units. With the selected separator, the units are complete positioned lines. It does not cut part of a line merely to manufacture an exact ten-character overlap.

The initial selected-separator audit, run with default whitespace stripping enabled, found 4,443 adjacent chunk pairs. An exact suffix-prefix text comparison found zero overlap for 3,115 pairs. This is expected when no complete line fits inside the ten-character allowance. Values above ten in that diagnostic can also arise from naturally repeated text, so the diagnostic must not be interpreted as an exact internal measurement of the splitter's retained units.

The correct description is therefore: a configured ten-character, best-effort overlap constrained to complete positioned lines. The dissertation and experiment logs must not claim that every adjacent pair contains exactly ten repeated characters.

## Oversized-unit rule

`CharacterTextSplitter` may emit a chunk larger than its configured size when one separator-delimited unit is already larger than that size. The audited source currently produces no such chunk. Production chunking must nevertheless check every result and stop with an error if any chunk exceeds 200 characters.

The baseline must not silently apply a second splitter because doing so would introduce an unrecorded algorithm. A future source with an oversized positioned line requires a new audited decision and configuration version.

## Production outputs

Chunking produces two deterministic files:

| Output | Purpose |
|---|---|
| `data/processed/corpus/chunks.jsonl` | Ordered retrieval chunks permitted to enter embedding and FAISS indexing |
| `data/processed/audit/chunking-summary.json` | Input identity, configuration identity, totals and chunk-export fingerprint |

These files may contain copyrighted textbook text and must remain excluded from Git. The summary is written last and must not contain a generation timestamp or its own fingerprint.

The reusable implementation is provided by `src/geotech_rag/corpus_chunking.py`. The verified production command is:

```bash
PYTHONPATH=src python -m geotech_rag.corpus_chunking \
    --config configs/chunking-config.json \
    --project-root .
```

The production run generated:

| Output | Records | Bytes | SHA-256 |
|---|---:|---:|---|
| `data/processed/corpus/chunks.jsonl` | 5,113 | 5,153,315 | `bce9702308a65b9e9ab3bcdb0781ad308d509b56904a95ec7bab4feffebca645` |
| `data/processed/audit/chunking-summary.json` | 1 | 1,532 | `d31c5ed3a3adca6d4f4f839e404c6582ef4419b385ffc5e5b49d253ac21f96d5` |

An independent audit parsed both input and output JSONL files without importing the production chunking module. It reconstructed every parent and chunk range, checked identifiers, metadata and bounding boxes, confirmed that all positioned lines were covered, found zero line-boundary problems and found zero leaked sentinels. A controlled overwrite reproduced both files byte for byte. Nine chunking-specific tests and the complete 42-test suite pass.

## Chunk-record contract

Every line of `chunks.jsonl` contains one JSON object with the following fields:

| Field | Type | Meaning |
|---|---|---|
| `chunk_schema_version` | string | Fixed value `1.0` |
| `chunk_id` | string | Stable parent identifier followed by `:chunk-NNNN` |
| `parent_record_id` | string | Identifier of the source retrieval-region record |
| `parent_record_schema_version` | string | Schema version of the parent record |
| `source_id` | string | Validated source identifier |
| `source_sha256` | string | Validated source fingerprint |
| `pdf_page_index` | integer | Zero-based PDF page index |
| `pdf_page_number` | integer | One-based PDF page number |
| `printed_page_number` | integer or null | Printed page number when available |
| `page_state` | string | Parent page state |
| `page_width` | number | Parent page width |
| `page_height` | number | Parent page height |
| `coordinate_origin` | string | Fixed value `top_left` |
| `coordinate_unit` | string | Fixed value `pdf_point` |
| `region_index` | integer | Parent retained-region index |
| `region_bbox` | array | Parent region bounding box |
| `chunk_index` | integer | Zero-based index within the parent record |
| `parent_character_start` | integer | Inclusive start offset in parent text |
| `parent_character_end` | integer | Exclusive end offset in parent text |
| `parent_line_start_index` | integer | Inclusive first positioned-line index |
| `parent_line_end_index` | integer | Exclusive final positioned-line index |
| `line_count` | integer | Number of complete positioned lines represented |
| `chunk_bbox` | array | Bounding box enclosing the represented lines |
| `text` | string | Ordered chunk text supplied to the embedding model |
| `character_count` | integer | Python `len(text)` |

Character ranges use half-open interval notation, `[start, end)`. Line ranges follow the same convention. Because overlap is allowed, adjacent chunks from the same parent may contain repeated line and character ranges.

The chunk text must equal the corresponding contiguous parent-text substring. Its start and end offsets must align with positioned-line text boundaries. Every parent line must be represented by at least one chunk, and chunk order must follow parent-record order followed by `chunk_index`.

## Stable identifiers

For the current source, a chunk identifier has the form:

```text
das-sobhan-2014-8e-si:pdf-0025:region-00:chunk-0000
```

Four decimal digits are used for the chunk index. Identifiers must be unique and deterministic. A configuration change that changes chunk boundaries is a new derived dataset and must produce a new audited fingerprint.

## Source-specific acceptance values

The selected candidate produced the following frozen audit values:

| Measure | Expected result |
|---|---:|
| Parent region records | 670 |
| Parent positioned lines | 39,934 |
| Parent characters | 843,350 |
| Retrieval chunks | 5,113 |
| Total emitted chunk characters, including overlap | 848,280 |
| Single-chunk parent records | 17 |
| Maximum chunks from one parent | 24 |
| Maximum chunk characters | 200 |
| Chunks over 200 characters | 0 |
| Chunks containing preserved formula controls | 2,119 |
| Positioned-line boundary problems | 0 |
| Positioned lines absent from all chunks | 0 |
| Leaked temporary `U+E000` sentinels | 0 |

The ordered audit projection of `parent_record_id`, `chunk_index` and `text` has SHA-256 `88aca7a9a60c21713f3e6d732cb17f6380776d1c2f0c06157f3414c018f09c4b`.

## Required validation failures

Production chunking must stop without publishing outputs if:

- the input path escapes the authorised `data/interim/corpus/` boundary;
- the input fingerprint, schema version or frozen counts differ;
- the configured temporary sentinel already occurs in parent text;
- an input record is empty, duplicated or out of order;
- a chunk is empty or exceeds 200 characters;
- a chunk cannot be mapped to one contiguous parent character range;
- a chunk boundary cuts through a positioned line;
- an embedded line newline is not restored or the temporary sentinel remains in output;
- a referenced line or parent record is missing;
- a chunk bounding box is non-finite or outside its parent region;
- any parent line is absent from all chunks;
- a chunk identifier is duplicated;
- ordered output or source-specific audit totals differ; or
- JSON serialisation contains a non-finite number.

## Deterministic serialisation

The chunk exporter must use UTF-8, one compact JSON object per JSONL line, sorted keys, `ensure_ascii=False`, `allow_nan=False` and a final newline. It must write through a temporary staging directory, calculate the completed chunk-file SHA-256, publish the summary last and require an explicit overwrite option before replacing an existing export.

## Later experimental use

The selected settings form the reconstructed baseline for RQ1 and must remain fixed during the generator-model comparison for RQ2. RQ3 may evaluate a different chunk size or the strict character-boundary candidate, but each sensitivity run must change one parameter at a time, use a separate configuration identifier and rebuild its own embeddings and FAISS index.
