# Baseline Replication Audit

## Purpose

This audit records which parts of the methodology reported by Tophel et al. (2025) can be reproduced directly and which parts require a documented replacement or implementation decision.

No value that is not reported by the paper should enter the baseline configuration without an explanation.

## Fixed research controls

- The retrieval corpus, evaluation questions and ground-truth answers must remain separate.
- Content from Das and Sobhan (2014) will form the retrieval corpus.
- The 391 textbook questions will form the wider evaluation bank.
- Official solution-manual answers will be used only as ground truth and must never enter the retrieval index.
- The Hard-20 questions are a focused subset of the wider evaluation bank.
- During model comparisons, the corpus, prompt, retrieval process and generation settings must remain fixed so that only the model version changes.

## Reported baseline components

| Component | Reported by Tophel et al. (2025) | Replication status |
|---|---|---|
| Environment | Dedicated Python virtual environment | Completed |
| Dependency versions | Exact package versions are not reported | Unresolved |
| API-key storage | Separate `constants.py` file | Replace with untracked `.env` for safer secret management |
| Input formats | PDF, DOCX and TXT document loaders | Reported; required formats will depend on the authorised corpus files |
| Text splitter | `CharacterTextSplitter` | Confirmed |
| Chunk size | `200` | Confirmed |
| Chunk overlap | `10` | Confirmed |
| Embeddings | OpenAI embeddings | Exact embedding model is not reported |
| Vector database | FAISS | Confirmed |
| Retrieval chain | LangChain conversational retrieval chain | Exact class and package version are not reported |
| Retrieval settings | `search_kwargs` are mentioned | Exact retrieval depth or `k` is not reported |
| Compared models | GPT-4 and Llama-3 | Exact model IDs and API endpoints are not reported |
| GPT-4 temperatures | `0.1`, `0.5` and `1` | Confirmed |
| `top_p` | `1` | Confirmed |
| Prompt | AI tutor prompt provided in Listing 1 | Available for later transcription and verification |
| Evaluation bank | 391 questions from Das and Sobhan (2014) | Confirmed |
| Focused benchmark | 20 challenging questions | Confirmed |
| Ground truth | Official textbook solution manual | Must remain outside the retrieval corpus |
| Evaluation dimensions | Accuracy, formula integration, explanation clarity and adaptability | Confirmed; operational scoring procedure still requires reconstruction |
| PDF extraction method | The exact loader and extraction behaviour are not reported | Positioned PyMuPDF selected after a comparative parser audit |
| Mixed-content filtering | The paper does not report filtering textbook problems or selected answers from a mixed-content PDF | Layout-aware page-coordinate exclusion is required to prevent evaluation leakage |
| Mathematical-symbol handling | Advanced formula integration is discussed as future development | No hand-built formula reconstruction is added to the replication baseline |

## Completed corpus-extraction decisions

The parser audit compared positioned PyMuPDF extraction, `pypdf` layout extraction, `pdfplumber` and Poppler `pdftotext`. Positioned PyMuPDF was selected because it preserves page coordinates and bounding-box provenance required for filtering the mixed-content textbook.

The layout investigation identified and validated the boundaries of the instructional content, chapter problem sections, references, selected answers and index. Problem statements and selected answers are excluded before any future chunking, embedding or FAISS index construction.

The retained extraction contains 9,825 unexpected control-character occurrences across 31 font/code groups. Glyph-aware analysis found that 20 of these groups contain multiple rendered glyph IDs. Visual validation confirmed that the same extracted font and control code can represent different mathematical symbols, including equality and addition signs or greater-than and multiplication signs.

A global control-character replacement would therefore corrupt mathematical content. No replacement mapping or hand-built formula reconstruction is applied in the replication baseline. Formula-aware reconstruction remains a possible separate extension whose effect must be evaluated independently.

The supporting implementation and evidence are recorded in:

- `notebooks/pdf_parser_audit.ipynb`, commit `855d5b9`;
- `notebooks/01_textbook_layout_exploration.ipynb`, commit `9f6c89e`.

The exact local source identity is frozen in `configs/corpus-manifest.json`. The manifest records the SHA-256 hash, byte size, PDF page count, bibliographic identity, permitted pipeline role and extraction decision without storing copyrighted textbook text.

Reusable enforcement is implemented in `src/geotech_rag/corpus_manifest.py`. Eight automated tests verify manifest structure, source integrity, parser version, fixed repository boundaries, leakage-prevention policies and the real local 770-page source.

The production record unit and deterministic export contracts are frozen in `docs/corpus-record-and-export-schema.md`. Notebook audits verify 670 non-empty retrieval-region records, complete reconstruction of cleaned page text, unique region identifiers and valid coordinates for all 40,370 positioned lines present before the final publisher-label exclusion.

The focused table and page-reference audit covers ordinary, continued, rotated, dense and side-by-side tables. It preserves the required captions and cross-references while documenting that the baseline does not reconstruct semantic table cells. Full-corpus re-extraction also verifies writing-direction provenance without changing any existing line. The exact publisher label occurs 436 times across 355 pages; its removal leaves 39,934 positioned lines, 843,350 characters and all 670 non-empty pages.

## Hard-20 question identifiers

The focused benchmark contains:

`2.3b`, `2.8c`, `2.9d`, `3.5c`, `3.6a`, `3.7d`, `3.7e`, `3.12c`, `4.5`, `5.2`, `6.6b`, `6.7a`, `6.10a`, `6.10b`, `7.9`, `11.4a`, `11.18a`, `12.8`, `12.10`, and `12.16`.

## Decision rule for missing details

For every detail not reported by the baseline paper:

1. record the missing information;
2. identify the currently available options;
3. select the closest technically defensible replacement;
4. explain how the replacement affects comparability;
5. freeze the decision in the baseline configuration before experiments begin.

## Sources

- Tophel, A., Chen, L., Hettiyadura, U., and Kodikara, J. (2025). *Towards an AI tutor for undergraduate geotechnical engineering*.
- Das, B. M., and Sobhan, K. (2014). *Principles of Geotechnical Engineering* (8th ed.).
- Approved dissertation proposal.
