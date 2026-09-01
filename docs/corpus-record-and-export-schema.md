# Production Corpus Record and Export Schema

## Decision status

- **Status:** Accepted for initial implementation
- **Decision date:** 2026-09-01
- **Record schema version:** `1.0`
- **Page-audit schema version:** `1.0`
- **Extraction-summary schema version:** `1.0`
- **Evidence source:** `notebooks/01_textbook_layout_exploration.ipynb`
- **Corpus manifest:** `configs/corpus-manifest.json`

## Purpose

This document defines the boundary between textbook extraction and later text chunking. It specifies which extracted content may become retrieval input, how its provenance must be preserved and how the resulting records must be exported.

The region records defined here are extraction records, not final embedding chunks. The later chunking stage may split their `text`, but every chunk must retain its parent record identifier and source-page provenance.

## Evidence from the audited source

The production record-unit audit produced the following fixed results:

| Measure | Result |
|---|---:|
| Source PDF pages | 770 |
| Fully retained pages and regions | 644 |
| Partially retained pages and regions | 32 |
| Fully excluded pages | 37 |
| Pages outside the knowledge scope | 57 |
| Retained regions before structural cleaning | 676 |
| Regions emptied by structural cleaning | 6 |
| Non-empty retrieval records | 670 |
| Pages containing multiple retained regions | 0 |
| Invalid line-to-region references | 0 |
| Page-text reconstruction mismatches | 0 |
| Duplicate candidate record keys | 0 |
| Positioned lines before publisher-label removal | 40,370 |
| Line-sequence mismatches after adding orientation metadata | 0 |
| Publisher copyright lines identified | 436 |
| Pages containing publisher copyright lines | 355 |
| Positioned lines after publisher-label removal | 39,934 |
| Characters after publisher-label removal | 843,350 |
| Pages emptied by publisher-label removal | 0 |
| Positioned lines checked | 40,370 |
| Strict geometry tolerance | 0.000001 points |
| Line-region comparison tolerance | 0.005001 points |
| Maximum observed boundary excess | 0.004616699218729536 points |
| Invalid line bounding boxes under the derived tolerance | 0 |

Heading bounding boxes are deterministically rounded to two decimal places, while positioned-line bounding boxes retain full parser precision. The line-region comparison tolerance is therefore derived as the maximum two-decimal rounding difference of `0.005` points plus the strict geometric tolerance of `0.000001` points. This validation tolerance does not change or enlarge extraction regions.

The six empty retained regions occur on PDF pages 87, 144, 370, 445, 509 and 727. They contained only structural text after the geometric exclusions were applied. They must remain represented in the page audit but must not become retrieval records.

A focused table audit additionally covered ordinary, continued, rotated, dense and side-by-side tables. Required table captions, continuation markers, figure references, equation references and page relationships remained present. Re-extraction with orientation metadata reproduced all 40,370 pre-removal positioned lines without changing their text, order, bounding boxes or provenance.

## Record-unit decision

One non-empty retained page region becomes one corpus record.

This unit was selected because it:

1. preserves geometric boundaries around excluded chapter problems;
2. prevents separated retained regions from being joined;
3. provides more useful context than an individual text line;
4. preserves page and bounding-box provenance;
5. remains independent from the later chunk-size decision; and
6. supports future sources that may contain multiple retained regions on one page.

For the current textbook, every retained page has exactly one retained region. The schema nevertheless keeps `region_index` because the production logic must support the general geometry contract rather than depend on this source-specific coincidence.

## Production exports

The extraction stage must produce three deterministic files under the manifest-authorised `data/interim/` directory.

| Output | Purpose | Expected source-specific count |
|---|---|---:|
| `data/interim/corpus/regions.jsonl` | Non-empty records permitted to enter chunking and retrieval | 670 records |
| `data/interim/audit/pages.jsonl` | Page-level geometry, cleaning and exclusion audit | 770 records |
| `data/interim/audit/extraction-summary.json` | Source identity, parser identity, totals and output fingerprints | 1 summary |

The derived exports may contain copyrighted source text and must not be tracked by Git.

## Retrieval-region record

Every line in `regions.jsonl` must contain one JSON object with the following fields.

| Field | Type | Meaning |
|---|---|---|
| `record_schema_version` | string | Fixed value `1.0` |
| `record_id` | string | Stable identifier for the source, PDF page and region |
| `source_id` | string | Source identifier copied from the validated corpus manifest |
| `source_sha256` | string | SHA-256 fingerprint copied from the validated manifest |
| `pdf_page_index` | integer | Zero-based source PDF page index |
| `pdf_page_number` | integer | One-based source PDF page number |
| `printed_page_number` | integer | Printed textbook page number |
| `page_state` | string | `fully_retained` or `partially_retained` |
| `page_width` | number | Source page width in PDF points |
| `page_height` | number | Source page height in PDF points |
| `region_index` | integer | Zero-based retained-region index within the page |
| `region_bbox` | array | Region coordinates `[x0, y0, x1, y1]` |
| `coordinate_unit` | string | Fixed value `pdf_point` |
| `coordinate_origin` | string | Fixed value `top_left` |
| `text` | string | Cleaned non-empty region text |
| `line_count` | integer | Number of retained positioned lines |
| `character_count` | integer | Python `len()` of the exact region text |
| `lines` | array | Ordered positioned-line records belonging to the region |

The stable record identifier must use:

```text
{source_id}:pdf-{pdf_page_number:04d}:region-{region_index:02d}
```

For example, region zero on PDF page 25 uses the location suffix:

```text
pdf-0025:region-00
```

The `region_bbox` uses the full page width and the retained vertical interval:

```text
[0.0, retained_y0, page_width, retained_y1]
```

The retained vertical interval follows the half-open form `[y0, y1)`.

## Positioned-line record

Each object in `lines` must contain:

| Field          | Type    | Meaning                                               |
| -------------- | ------- | ----------------------------------------------------- |
| `line_index`   | integer | Zero-based line order within the region               |
| `text`         | string  | Extracted non-empty line text                         |
| `bbox`         | array   | Line coordinates `[x0, y0, x1, y1]`                   |
| `block_number` | integer | PyMuPDF source block number                           |
| `line_number`  | integer | Line number within its source block                   |
| `span_count`   | integer | Number of source spans joined to reconstruct the line |
| `direction` | array | Two-number PyMuPDF writing-direction vector `[dx, dy]` |
| `writing_mode` | integer | PyMuPDF writing-mode value |

`line_index` must be contiguous from zero. The region-level `text` must equal the line texts joined in order with exactly one newline character.

The two `direction` values must be finite numbers and must be serialised without coordinate rounding. The current source uses writing mode zero throughout, but `writing_mode` remains required so that the exported record preserves the parser's complete orientation contract.

The existing line-level `region_index` is not repeated inside the exported line object because the parent region record already provides it. Removing the duplicate prevents conflicting region identifiers.

## Table and page-reference preservation

The baseline preserves tables as ordered positioned lines with text, bounding boxes, writing direction, writing mode and page provenance. It does not reconstruct semantic rows, columns, merged cells or table relationships that are not explicitly provided by the selected extraction method.

The focused audit verified required markers on seven representative PDF pages covering ordinary, continued, rotated, dense and side-by-side tables. All selected table captions, continuation markers, figure references and equation references remained present.

PyMuPDF's automatic table detector found no table on six of the seven representative pages and returned only a 1-by-2 structure on the remaining page. Its output is therefore not accepted as a reliable table reconstruction for this source.

The optional `pymupdf_layout` extension is not part of the reconstructed baseline. Introducing it would change the extraction method and requires a separate comparative audit. It may be evaluated later as an enhanced method if retrieval results show that semantic table reconstruction is necessary.

## Page-audit record

Every line in `pages.jsonl` must contain one JSON object. All 770 source PDF pages must be represented, including fully excluded and out-of-scope pages.

| Field | Type | Meaning |
|---|---|---|
| `page_audit_schema_version` | string | Fixed value `1.0` |
| `source_id` | string | Source identifier from the validated manifest |
| `source_sha256` | string | Validated source fingerprint |
| `pdf_page_index` | integer | Zero-based PDF page index |
| `pdf_page_number` | integer | One-based PDF page number |
| `printed_page_number` | integer or null | Printed textbook page number, or null before printed page 1 |
| `page_width` | number | Page width in PDF points |
| `page_height` | number | Page height in PDF points |
| `coordinate_unit` | string | Fixed value `pdf_point` |
| `coordinate_origin` | string | Fixed value `top_left` |
| `page_state` | string | Final geometric page state |
| `excluded_y_intervals` | array | Ordered excluded half-open vertical intervals |
| `retained_y_intervals` | array | Ordered retained half-open vertical intervals |
| `raw_line_count` | integer | Positioned lines before structural cleaning |
| `raw_character_count` | integer | Characters before structural cleaning |
| `cleaned_line_count` | integer | Lines after structural cleaning |
| `cleaned_character_count` | integer | Characters after structural cleaning |
| `removed_structural_line_count` | integer | Number of removed structural lines |
| `removed_structural_lines` | array | Removed lines with their removal reasons |
| `emitted_region_count` | integer | Retrieval records emitted from this page |
| `emitted_record_ids` | array | Ordered identifiers of emitted region records |

The permitted `page_state` values are:

- `fully_retained`;
- `partially_retained`;
- `fully_excluded`; and
- `outside_knowledge_scope`.

Each removed structural-line object must contain `text`, `bbox`, `region_index`, `block_number`, `line_number`, `span_count`, `direction`, `writing_mode` and `removal_reason`.

The permitted removal reasons are `remove_running_header`, `remove_bottom_page_number` and `remove_publisher_copyright`.

A page with no emitted retrieval record remains present in this audit file. The audit export must never be used as retrieval input.

## Extraction summary

`extraction-summary.json` must contain one JSON object with the following fields.

| Field | Type | Meaning |
|---|---|---|
| `extraction_summary_schema_version` | string | Fixed value `1.0` |
| `manifest_relative_path` | string | Repository-relative corpus-manifest path |
| `source_id` | string | Validated source identifier |
| `source_sha256` | string | Validated source fingerprint |
| `source_size_bytes` | integer | Validated source byte size |
| `pdf_page_count` | integer | Validated PDF page count |
| `parser_library` | string | Fixed value `PyMuPDF` |
| `parser_version` | string | Validated installed package version |
| `record_schema_version` | string | Region-record schema version |
| `page_audit_schema_version` | string | Page-audit schema version |
| `page_state_counts` | object | Counts for all four page states |
| `structural_removal_counts` | object | Counts for all three removal reasons |
| `line_orientation_counts` | array | Ordered direction, writing-mode and line-count records |
| `retrieval_record_count` | integer | Number of emitted region records |
| `retrieval_line_count` | integer | Total retained positioned lines |
| `retrieval_character_count` | integer | Total retained characters |
| `empty_retained_region_pdf_pages` | array | PDF pages whose retained regions became empty |
| `regions_relative_path` | string | Repository-relative region-export path |
| `regions_sha256` | string | SHA-256 of the completed region export |
| `page_audit_relative_path` | string | Repository-relative page-audit path |
| `page_audit_sha256` | string | SHA-256 of the completed page-audit export |

Each object in `line_orientation_counts` must contain `direction`, `writing_mode` and `line_count`. Records must be ordered by the first direction value, the second direction value and then writing mode. Their `line_count` values must sum to `retrieval_line_count`.

For the frozen source, `retrieval_record_count` must equal `670`, `retrieval_line_count` must equal `39934`, `retrieval_character_count` must equal `843350` and `empty_retained_region_pdf_pages` must equal `[87, 144, 370, 445, 509, 727]`.

The summary must not contain a generation timestamp because that would make otherwise identical exports differ between runs. The summary cannot contain its own fingerprint because that would create a self-referential value.

## Text-preservation rules

The baseline export must:

1. remove only ordinary edge whitespace through `trim_text_edges`;
2. remove only the validated running headers, printed page numbers and the exact publisher line `© Cengage Learning 2014`;
3. preserve line order, bounding boxes, writing direction and writing mode;
4. join spans without inserting artificial characters;
5. preserve unusual formula control characters;
6. perform no Unicode normalisation;
7. perform no formula reconstruction;
8. perform no dehyphenation;
9. perform no semantic rewriting; and
10. exclude image blocks from the text-only baseline.

JSON escaping of control characters changes only their file representation. Reading the JSON must recover the same Python string values.

## Deterministic serialisation

The exporters must:

* use UTF-8;
* write one object per line in each JSONL file;
* order region records by `pdf_page_index` and then `region_index`;
* order page-audit records by `pdf_page_index`;
* reject `NaN` and infinite numeric values;
* use stable JSON key ordering;
* terminate every JSONL record with `\n`;
* avoid timestamps and random identifiers; and
* calculate output fingerprints only after the files are complete.

## Required validation

Production extraction must fail if any of the following occurs:

* the source fails corpus-manifest validation;
* a page identifier does not match its zero-based and one-based relationship;
* a printed page number does not match the validated offset;
* a region lies outside its physical page;
* a retained line refers to a missing region;
* a line bounding box lies outside its retained region by more than the derived `0.005001`-point line-region tolerance;
* a line direction is not an array of two finite numbers;
* a line writing mode is not an integer;
* the line-orientation counts do not sum to the retrieval line count;
* the exact publisher line `© Cengage Learning 2014` remains in retrieval text;
* a region identifier is duplicated;
* region text differs from its ordered line reconstruction;
* a retrieval record is empty;
* a retrieval record originates from an excluded page state;
* a required source-specific count differs from the audited baseline; or
* the page audit does not contain exactly 770 ordered records.

## Leakage boundary

Only `regions.jsonl` may enter the chunking stage. The raw PDF, page-audit export, evaluation questions, ground truth, chapter problems, selected answers and Index must never be passed to the retrieval indexer.
