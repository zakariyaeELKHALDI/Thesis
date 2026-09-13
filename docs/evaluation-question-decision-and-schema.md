# Evaluation question decision and schema

## Decision

The retrieval benchmark uses the 20 difficult textbook questions named by Tophel et al. (2025). The questions are stored locally as text-only, standalone queries in `data/raw/evaluation/questions/benchmark-questions.jsonl`.

The raw question file is intentionally ignored by Git. Its structure, expected identifiers, provenance rules and fingerprints are frozen in `configs/evaluation-question-config.json`, which may be committed.

Retrieval depth `k` remains unresolved. It must be selected through a separate retrieval experiment using these questions, not by adopting a library default.

## Why this stage is separate

The corpus index contains instructional textbook content only. Evaluation questions are runtime queries and must never be added to that index. Ground-truth answers remain unavailable to question embedding, retrieval and answer generation. They may be opened only by the later scoring stage after an answer has been generated.

This separation supports the dissertation methodology marks because it makes the benchmark construction, leakage controls and reconstruction assumptions explicit. It also prepares auditable retrieval results for the results and interpretation section.

## Evidence and reconstruction boundary

The base paper reports a 391-question bank and identifies a difficult 20-question subset, but it does not publish the complete typed query strings. The 20 identifiers are:

`2.3b`, `2.8c`, `2.9d`, `3.5c`, `3.6a`, `3.7d`, `3.7e`, `3.12c`, `4.5`, `5.2`, `6.6b`, `6.7a`, `6.10a`, `6.10b`, `7.9`, `11.4a`, `11.18a`, `12.8`, `12.10`, `12.16`.

The queries were therefore manually reconstructed from the cited eighth-edition SI textbook. Cross-references such as "repeat part b" were expanded so every text-only query can be embedded and interpreted independently. Tables were linearised without adding solutions.

This is a close, documented reconstruction, not a claim that the exact character strings used by the authors are known.

## Problem 6.6 source discrepancy

Tophel et al. identify one benchmark item as `6.6b`. The inspected textbook prints Problem 6.6 as two consecutive questions without visible `(a)` and `(b)` labels. The reconstructed record preserves the published benchmark identifier `6.6b` and maps it to the second question, which asks for the number of 20-ton truckloads. The fill-volume and soil-property values remain in the query because they are needed to solve that task.

This discrepancy is represented by `source_label_status = "benchmark_label_not_printed_in_textbook"`; it is not silently corrected.

## Frozen record schema

Each JSON Lines record contains exactly one runtime evaluation question.

| Field | Meaning |
|---|---|
| `question_schema_version` | Record schema version, fixed at `1.0` |
| `benchmark_dataset_id` | Stable identifier for this reconstructed 20-question set |
| `position` | Zero-based order matching the paper's published identifier list |
| `question_id` | Identifier reported by the base paper |
| `textbook_problem` | Printed parent-problem number |
| `textbook_part` | Printed or benchmark-inferred part label, or `null` |
| `chapter` and `topic` | Textbook subject provenance |
| `pdf_page_numbers` | One-based page number or page span in the inspected PDF |
| `printed_page_numbers` | Page number or page span printed in the textbook |
| `query_text` | Standalone text-only retrieval and generation input |
| `query_text_sha256` | UTF-8 fingerprint of `query_text` |
| `contains_table` | Whether a source table was linearised into text |
| `contains_figure` | Whether the query depends on a figure, fixed to `false` here |
| `transcription_mode` | Declares manual text-only reconstruction |
| `source_label_status` | Whether the benchmark label matches the printed textbook label |
| `source_note` | Explicit expansion, normalisation or discrepancy note |
| `ground_truth_available_during_generation` | Leakage guard, fixed to `false` |

## Frozen audit values

| Check | Expected value |
|---|---:|
| Question records | 20 |
| Tables linearised | 6 |
| Figure-dependent questions | 0 |
| Source-label discrepancies | 1 |
| JSONL SHA-256 | `ec7c151ba709eb6e55d37290dc8fb349a5def8b204f439b8e354f71d0b97d599` |
| Canonical question projection SHA-256 | `d785ffa3164d330f689512e4b52794a6316fa94596287bbb60ddf1378e27b343` |

Chapter counts are 3 for Chapter 2, 5 for Chapter 3, 1 for Chapter 4, 1 for Chapter 5, 4 for Chapter 6, 1 for Chapter 7, 2 for Chapter 11 and 3 for Chapter 12.

## Validation requirements

A reusable loader must reject the dataset if:

1. the JSONL byte fingerprint differs from the configuration;
2. the question count, order or identifier list differs;
3. positions are not unique and contiguous from 0 to 19;
4. a query is empty or its individual fingerprint is wrong;
5. a record contains a figure dependency;
6. any record makes ground truth available during retrieval or generation;
7. the chapter, table or discrepancy audit differs from the frozen values;
8. the question file is confused with corpus-index input.

## Next experiment

After the loader and tests reproduce this contract, the existing FAISS index can be queried with the same embedding model. Candidate values of `k` should then be compared using retrieval evidence that is independent of the hidden solution answers. Ground truth remains closed until the answer-generation step has finished.
