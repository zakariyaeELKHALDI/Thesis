# Pipeline Architecture and Dependency Strategy

## Decision record

- **Status:** Accepted for initial implementation, subject to the verification gates defined below
- **Date:** 2026-07-27
- **Last reviewed:** 2026-09-02
- **Scope:** Baseline reconstruction, model comparison and parameter-sensitivity experiments
- **Related documents:** `docs/baseline-replication-audit.md`; `docs/corpus-record-and-export-schema.md`

## Purpose

This document records the reasoning used to select the software architecture and dependency strategy for the RAG-based geotechnical engineering tutor.

The purpose is not only to identify which technologies will be used. It is also to explain:

- what is directly reported by Tophel et al. (2025);
- what remains missing from the baseline paper;
- which alternatives were considered;
- why the selected architecture is the most defensible option;
- how the decision affects baseline comparability;
- how the architecture supports the approved research questions;
- and how the implementation will be verified before experiments begin.

This decision must be recorded before installing the main RAG dependencies so that the software choices follow the research design rather than determining it afterwards.

## Decision classification

| Decision | Classification | Reason |
|---|---|---|
| Use `CharacterTextSplitter` with chunk size `200` and overlap `10` in the reconstructed baseline | Direct replication | These details are explicitly reported in Table 2 of Tophel et al. (2025) |
| Use OpenAI embeddings and FAISS in the reconstructed baseline | Direct replication at component level | Both components are reported, but their exact configurations are not |
| Build one explicit modular RAG pipeline | Reconstructed baseline decision | The paper reports a conversational retrieval chain but does not identify its exact class or internal implementation |
| Use current component-specific LangChain packages | Practical adjustment | The LangChain package structure has changed since the baseline implementation |
| Use `faiss-cpu` directly rather than the archived `langchain-community` wrapper | Reconstructed baseline decision and practical adjustment | The paper reports FAISS but does not state which wrapper was used |
| Use `.env` rather than `constants.py` for API keys | Practical security adjustment | This preserves the paper's separation of credentials while preventing secrets from entering Git |
| Use positioned PyMuPDF for textbook extraction | Reconstructed baseline decision | The paper does not identify its exact PDF loader, while PyMuPDF preserves the geometry required for mixed-content filtering and provenance |
| Exclude problem and selected-answer regions by page coordinates | Data-leakage prevention decision | The textbook combines instructional content with evaluation and ground-truth material in the same PDF |
| Leave ambiguous mathematical control characters unreconstructed in the baseline | Baseline-scope decision | Glyph-aware evidence shows that global control-code replacement would corrupt mathematical symbols |
| Run every benchmark question in a clean session | Reconstructed experimental-control decision | The paper does not explain its conversation-history procedure, and shared history would create order effects |
| Allow conversation history only in a separate interactive tutor mode | Thesis implementation decision | A working tutor should support follow-up interaction, but this must not affect controlled benchmarking |
| Change only the generator model during the main model comparison | Direct requirement from the approved proposal | This is required for a fair comparison under Research Question 2 |
| Change one parameter at a time during sensitivity analysis | Direct requirement from the approved proposal | This isolates the effect of each parameter under Research Question 3 |

## Evidence from the baseline paper

Table 2 of Tophel et al. (2025), on PDF page 8, reports the following implementation stages:

1. creation of a dedicated virtual environment;
2. installation of document-processing, embedding and conversational-model libraries;
3. separate API-key storage;
4. loading PDF, DOCX and TXT files;
5. document processing with `CharacterTextSplitter`;
6. `chunk_size = 200`;
7. `chunk_overlap = 10`;
8. OpenAI embeddings;
9. FAISS similarity search;
10. a conversational retrieval chain using GPT-4 and Llama-3;
11. generation and formatting of chatbot responses.

The paper also reports temperature and `search_kwargs` as chain hyperparameters. However, it does not report:

- the Python version;
- dependency versions;
- the exact LangChain chain class;
- the exact text-splitter separator or length function;
- the exact OpenAI embedding model;
- whether embeddings were normalised;
- the FAISS index type;
- the similarity metric;
- the retrieval depth `k`;
- the exact structure of `search_kwargs`;
- the exact GPT-4 and Llama-3 model identifiers;
- the Llama-3 provider or API endpoint;
- response-token limits;
- random seeds, where supported;
- the conversation-history implementation;
- or the complete logging and evaluation procedure.

Therefore, this thesis can reproduce the reported methodology, but it cannot claim to reproduce the authors' exact code or software environment.

## Requirements from the approved proposal

The approved proposal requires:

- a working RAG-based tutor prototype;
- textbook content stored and searched through FAISS;
- a LangChain-style pipeline connecting retrieval with LLMs;
- application of the baseline methodology;
- comparison of newer LLM versions within the same system design;
- a model comparison in which the dataset, prompt and pipeline remain consistent;
- sensitivity analysis covering parameters such as temperature, retrieval depth and chunk size;
- and optional extensions addressing formula reliability, multimodal capability or reasoning consistency.

The selected architecture must therefore support both baseline reconstruction and controlled extensions without changing the complete system between experiments.

## Assessment alignment

The CDS dissertation guidance in the module handbook requires the Approach section to:

- explain design choices, assumptions and methodological rationale;
- specify tools, frameworks, datasets and environments;
- include sufficient reproducibility information;
- provide detailed technical implementation information;
- cover architectural and programming decisions;
- and explain technical difficulties and how they were addressed.

Recording this decision before implementation creates evidence for these requirements and provides material for the dissertation's Approach, Evaluation and Limitations sections.

## Relationship to the research questions

| Research question | Architectural requirement |
|---|---|
| RQ1: Apply the methodology of Tophel et al. | Preserve reported components while documenting every reconstructed detail |
| RQ2: Compare newer LLM versions | Freeze the corpus, embeddings, index, retrieval, prompt and generation settings so only the model changes |
| RQ3: Test controlled parameter changes | Make chunking, retrieval and generation settings explicit and configurable |
| RQ4: Address baseline limitations | Keep optional validation or multimodal components separate from the reconstructed baseline |

## Architecture options considered

| Option | Baseline similarity | Transparency and control | Maintainability | Workload and experimental risk | Decision |
|---|---|---|---|---|---|
| Use only `ConversationalRetrievalChain` from `langchain-classic` | Superficially similar to the phrase used in the paper, but the exact class is not confirmed | Some behaviour is handled internally and may be harder to inspect | The class is deprecated | Moderate workload, but risks treating an assumption as a direct replication | Rejected as the main architecture |
| Build one explicit modular RAG pipeline using current LangChain components and direct FAISS | Preserves every reported component while making missing decisions visible | High transparency over chunking, retrieval, prompts and logging | Uses current component packages | Manageable workload and strongest experimental control | Selected |
| Build both a classic chain and an explicit pipeline | Could provide an additional architectural comparison | High, but results would introduce another changing variable | Two pipelines must be maintained | Doubles implementation and validation work without being required by the proposal | Postponed unless later evidence or time justifies it |
| Build the complete system without any LangChain components | FAISS and RAG could still be reproduced | Maximum low-level control | Requires more custom code | Weaker alignment with the approved LangChain-style design and unnecessary reimplementation | Rejected |

## Selected architecture

The project will use one explicit, modular RAG pipeline for the reconstructed baseline and the main experiments.

The pipeline will expose each stage separately:

1. authorised source ingestion;
2. extraction and validation of textbook content;
3. document normalisation;
4. document chunking;
5. embedding generation;
6. FAISS index construction;
7. question embedding;
8. top-`k` retrieval;
9. prompt construction;
10. model generation;
11. provenance and experiment logging;
12. separate evaluation against ground truth.


Ground-truth solutions must remain unavailable to corpus chunking, corpus embedding, FAISS index construction, retrieval and model generation. They may be loaded only by the separate evaluation stage after the corresponding model answer has been generated.

During corpus preparation, problem statements, their question parts and selected answers may be inspected only to detect and exclude their regions. They must never become retained instructional content or be passed to corpus chunking, corpus embedding, FAISS index construction or the retrieved context.

Problem statements may later be processed through a separate evaluation-only pipeline and supplied as runtime retrieval queries and generation inputs. Selected answers may be processed only as ground truth and must remain unavailable until after the corresponding model answer has been generated.

## Mixed-content textbook filtering

The Das and Sobhan textbook is a mixed-content source: instructional material, end-of-chapter problems and selected answers are stored in the same PDF. The original PDF will remain unchanged for provenance, but it must never be passed directly to chunking or index construction.

A layout-aware preprocessing stage will identify the exact `Problems`, `References`, `Answers to Selected Problems` and `Index` heading blocks and create page-coordinate exclusion regions. For every chapter, all content from the `Problems` heading up to, but not including, the `References` heading will be excluded. This includes parent problems, lettered question parts, critical-thinking problems, figures, tables and content continuing across pages. The complete `Answers to Selected Problems` section will also be excluded.

Content before a mid-page `Problems` heading and content from a mid-page `References` heading onward will be retained. Only the filtered instructional output under `data/interim/` may enter chunking, embedding and FAISS index construction.

The preprocessing stage must stop with an error if the expected boundary headings are missing, duplicated or out of order. Every retained element must preserve its source PDF page and bounding-box provenance. An automated audit must confirm that no retained element overlaps an excluded region before the index is built.

### Completed parser and layout audit

The comparative audit tested positioned PyMuPDF extraction, `pypdf` layout extraction, `pdfplumber` and Poppler `pdftotext`. Positioned PyMuPDF was selected for production because it retained the page and bounding-box geometry required by the exclusion rules and provenance design.

The verified environment uses PyMuPDF `1.28.2`. The locked `pypdf 6.16.2` and `pdfplumber 0.11.10` packages remain available to reproduce the comparative audit, but they are not selected as production corpus extractors. Poppler `pdftotext` remains a system-level audit comparator.

The layout investigation validated the textbook heading boundaries and retained regions before any corpus chunking. It also matched all 9,825 retained unexpected control characters to rendered glyph IDs. Twenty of the 31 font/code groups contained multiple glyph IDs, and visual examples confirmed that identical extracted font/code combinations can represent different mathematical symbols.

Global control-character replacement is therefore rejected. The reconstructed baseline will document the extracted mathematical-font limitation without adding hand-built formula reconstruction. Any formula-aware reconstruction must be implemented and evaluated later as a separate experimental extension.

The supporting evidence is stored in `notebooks/pdf_parser_audit.ipynb` at commit `855d5b9` and `notebooks/01_textbook_layout_exploration.ipynb` at commit `9f6c89e`.

## Repository data boundaries

The repository separates source documents, benchmark inputs, reference answers and generated results so that evaluation material cannot be incorporated into the retrieval corpus.

| Path | Role | Permitted pipeline use |
|---|---|---|
| `data/raw/corpus/` | Authorised textbook source files | The only raw-data directory permitted as input to source ingestion, extraction, chunking, corpus embedding and FAISS index construction |
| `data/raw/evaluation/questions/` | Wider question bank and focused Hard-20 benchmark questions | May be loaded only as runtime evaluation queries after index construction; must never be treated as corpus documents or stored in the FAISS index |
| `data/raw/ground_truth/` | Official solution-manual reference answers | May be accessed only by the separate evaluation stage after an answer has been generated; must never enter retrieval or generation |
| `results/runs/` | Generated answers, retrieved-chunk records, configurations, timings and provenance | Output location only; previous run artefacts must not become inputs to the same controlled benchmark |

The following enforcement rules apply:

1. Corpus ingestion must be explicitly restricted to `data/raw/corpus/`.
2. A recursive corpus loader must never be pointed at the complete `data/raw/` directory.
3. Evaluation questions may be embedded as retrieval queries at runtime, but they must not be added to the corpus index.
4. Ground-truth answers must remain unavailable to retrieval and generation and may be loaded only after model-answer generation.
5. Actual source, question and ground-truth files must remain ignored by Git; only their structural `.gitkeep` placeholders may be tracked.
6. An automated leakage-prevention test must verify these boundaries before the experimental pipeline is accepted.


## Why an explicit pipeline was selected

The baseline paper describes a conversational retrieval chain but does not prove that the authors used the class named `ConversationalRetrievalChain`.

Using that class would therefore be an assumption rather than a direct replication.

An explicit pipeline is more defensible because it allows the thesis to record and verify:

- the exact retrieved chunk identifiers;
- the rank and similarity score of each chunk;
- the retrieval depth;
- the embedding model;
- the FAISS index and metric;
- the complete prompt version;
- the model identifier;
- generation parameters;
- the generated answer;
- token usage and latency where available;
- and errors or retries.

This information is necessary to determine whether performance differences came from the LLM or from another part of the system.

## Benchmark mode and interactive mode

The system will distinguish between two modes.

| Mode | Conversation history | Purpose |
|---|---|---|
| Benchmark mode | Disabled between evaluation questions | Controlled experiments and reproducible comparison |
| Interactive tutor mode | May be enabled for follow-up questions | Demonstration of the working tutor prototype |

Every benchmark question will start from a clean model context.

This prevents:

- information from an earlier question influencing a later answer;
- question-order effects;
- accidental transfer of retrieved content between questions;
- and differences caused by providers handling chat history differently.

If conversational memory is implemented for the prototype, it will be tested separately and will not be used in the main benchmark unless a later experiment explicitly studies it.

## Dependency strategy

The project will continue using one Python `3.12.3` virtual environment.

`requirements.in` will record intentionally selected direct dependencies. `requirements-lock.txt` will record the exact resolved environment used for the experiments.

The baseline paper does not report its dependency versions. Current maintained versions will therefore be used and locked rather than presenting them as exact baseline versions.

### Initial RAG dependency group

| Package | Intended use | Decision |
|---|---|---|
| `langchain-core` | Provider-independent documents, messages, prompts and interfaces used directly by the implementation | Include |
| `langchain-openai` | Current OpenAI chat-model and embedding integrations | Include for the OpenAI implementation |
| `langchain-text-splitters` | Provides the reported `CharacterTextSplitter` | Include |
| `faiss-cpu` | Provides FAISS vector indexing and similarity search | Include and use directly |
| `numpy` | Provides the array representation passed directly to FAISS | Include because project code will import it |
| `python-dotenv` | Loads API credentials from the untracked `.env` file | Include |
| `langchain-classic` | Provides legacy chains such as `ConversationalRetrievalChain` | Exclude from the main dependency group |
| `langchain-community` | Provides older community wrappers, including a FAISS wrapper | Exclude because it is archived and direct FAISS gives clearer control |
| `pymupdf` | Provides positioned textbook extraction, page geometry and glyph-level diagnostics | Include as the production extractor; selected after the comparative parser audit |
| `pypdf` and `pdfplumber` | Provide alternative extraction outputs used by the parser audit | Retain as audit dependencies, but do not use for baseline corpus production |
| `pymupdf_layout` | Optional enhanced page-layout and table reconstruction | Exclude from the baseline; evaluate separately only if retrieval results justify an enhanced table method |
| Additional provider SDKs | Connect to non-OpenAI models | Defer until the comparison models and endpoints are selected |
| Evaluation libraries | Statistical processing and result tables | Add later when the scoring procedure is frozen |
| `pytest` | Automated unit and integration tests | Included with the first reusable module and locked at version `9.1.1` |

### Why direct FAISS was selected

The paper explicitly reports FAISS but does not report whether it used the LangChain FAISS wrapper or direct FAISS bindings.

Using `faiss-cpu` directly allows the project to control and document:

- index type;
- vector dimensionality;
- similarity metric;
- vector normalisation;
- mapping between index positions and chunk identifiers;
- retrieval depth;
- ranking;
- similarity scores;
- index persistence;
- and index reloading.

This requires slightly more implementation code, but the additional code represents meaningful technical work and improves reproducibility.

## Remaining unresolved decisions

The authorised source has been identified as Das and Sobhan (2014), *Principles of Geotechnical Engineering*, 8th SI edition. Positioned PyMuPDF has also been selected as the production extraction method. The source identity and these decisions are frozen in `configs/corpus-manifest.json`. Reusable validation now enforces the manifest, source fingerprint and fixed repository boundaries in `src/geotech_rag/corpus_manifest.py`. The production region-record unit, page-audit contract and deterministic export summary are now frozen in `docs/corpus-record-and-export-schema.md`, based on the completed notebook audits.

The focused table audit confirms preservation of required captions, continuation markers and page references but does not support semantic row-and-column reconstruction. The baseline will therefore export ordered positioned lines with bounding boxes, writing direction and writing mode. It will also remove only the exact repeated publisher label identified by the full-corpus audit.

The production extraction and deterministic export chain is now implemented, tested against the validated source and verified through byte-for-byte export regeneration.

The following values remain unresolved before the baseline configuration can be frozen:

1. text-normalisation rules beyond the validated edge trimming and structural exclusions;
2. text-splitter separator and length function;
3. handling of chunks that exceed the intended size;
4. exact OpenAI embedding model;
5. embedding normalisation;
6. FAISS index type and similarity metric;
7. retrieval depth `k`;
8. exact baseline prompt transcription;
9. handling of insufficient retrieved evidence;
10. generator model identifiers and API endpoints;
11. maximum response length;
12. retry and error-handling rules;
13. repeated-run strategy for nondeterministic outputs;
14. evaluation rubric and scoring procedure;
15. storage format for experiment results;
16. selected models for the newer-model comparison;
17. optional enhanced formula or unit-consistency extension.

Each item must receive its own evidence-based decision or be grouped with technically related items.

## Experimental-control rules

The following controls apply to the main experiments:

1. Retrieval content and ground-truth solutions must remain separate.
2. The same processed corpus must be used across compared models.
3. The same embedding model and saved FAISS index must be used across compared models.
4. The same evaluation questions must be used.
5. The same prompt version must be used.
6. The same retrieval settings must be used.
7. The same generation settings must be used where provider APIs allow equivalent settings.
8. Every benchmark question must use a clean session.
9. During the main model comparison, only the generator model may change.
10. Provider-specific differences that cannot be controlled must be recorded as limitations.
11. During sensitivity analysis, one selected parameter must change at a time.
12. Optional RQ4 extensions must be evaluated against the frozen baseline rather than silently added to it.

## Reproducibility and logging requirements

Each experimental record should contain, where available:

- run identifier;
- experiment configuration identifier;
- timestamp;
- question identifier;
- dataset version;
- corpus-processing version;
- embedding model;
- FAISS index identifier;
- retrieval configuration;
- retrieved chunk identifiers;
- retrieved ranks and scores;
- prompt version or hash;
- model provider;
- exact model identifier returned or requested;
- temperature;
- `top_p`;
- token limit;
- generated answer;
- token usage;
- latency;
- retry count;
- error status;
- and evaluation scores.

API keys, personal credentials and copyrighted source text must not be written into experiment logs.

## Effect on baseline comparability

The selected architecture provides a methodology-level reconstruction, not an exact software replication.

Comparability is strengthened because the project preserves the paper's reported:

- splitter class;
- baseline chunk size;
- baseline overlap;
- OpenAI embeddings;
- FAISS retrieval;
- tutor prompt;
- evaluation foundation;
- focused Hard-20 benchmark;
- temperature settings;
- and evaluation dimensions.

Comparability is limited because the paper does not provide enough information to reproduce its exact:

- dependency environment;
- chain implementation;
- embedding configuration;
- FAISS configuration;
- retrieval settings;
- model endpoints;
- or conversation-history behaviour.

These limitations will be reported openly. No reconstructed value will be described as if it was directly reported by the paper.

## Verification gates

The architecture will be accepted for experiments only after the following checks pass:

1. dependency resolution dry run under Python `3.12.3`;
2. installation and import smoke test;
3. `CharacterTextSplitter` behaviour test;
4. FAISS index creation and search test using synthetic text;
5. FAISS save-and-reload consistency test;
6. chunk-identifier and metadata test;
7. retrieval logging test;
8. corpus extraction-quality audit, including formulas and tables;
9. check that ground-truth answers cannot enter the retrieval index;
10. one-question end-to-end RAG smoke test;
11. test showing that benchmark questions start with clean state;
12. exact dependency lock regeneration;
13. frozen baseline configuration;
14. Git checkpoint containing the verified implementation and documentation.

The corpus-manifest boundary and source-integrity rules are implemented in `src/geotech_rag/corpus_manifest.py` and verified against the local 770-page source. The complete production chain is now implemented in reusable geometry, boundary, positioned-extraction and deterministic-export modules. The 33-test automated suite includes real-source integration checks and reproduces the frozen totals of 670 retrieval records, 770 page audits, 39,934 positioned lines and 843,350 characters. The production run created all three manifest-authorised interim exports, an independent audit verified their contents and fingerprints, and a controlled regeneration reproduced all three files byte for byte. Extraction-quality gate 8 is therefore satisfied. The copyrighted derived exports remain excluded from Git.

A successful package installation alone will not be treated as proof that the RAG system works correctly.

## Intended dissertation use

This decision record will support:

- the architecture and design-rationale discussion in the Approach chapter;
- the experimental controls in the Evaluation chapter;
- the difference between replication and reconstruction in the Limitations section;
- the technical implementation details in the repository;
- and additional configuration or verification evidence in the appendices.

## Sources

- Tophel, A., Chen, L., Hettiyadura, U., and Kodikara, J. (2025). *Towards an AI tutor for undergraduate geotechnical engineering*. See PDF pages 5, 8 and 13.
- Approved dissertation proposal, pages 1-2.
- Gisma Dissertation Module Handbook, CDS dissertation structure and Approach guidance, PDF page 23.
- LangChain. *LangChain v1 migration guide*. https://docs.langchain.com/oss/python/migrate/langchain-v1
- LangChain. *ConversationalRetrievalChain reference*. https://reference.langchain.com/python/langchain-classic/chains/conversational_retrieval/base/ConversationalRetrievalChain
- LangChain. *OpenAI integrations*. https://docs.langchain.com/oss/python/integrations/providers/openai
- Python Package Index. *langchain-community*. https://pypi.org/project/langchain-community/
- Python Package Index. *langchain-core*. https://pypi.org/project/langchain-core/
- Python Package Index. *langchain-openai*. https://pypi.org/project/langchain-openai/
- Python Package Index. *langchain-text-splitters*. https://pypi.org/project/langchain-text-splitters/
- Python Package Index. *faiss-cpu*. https://pypi.org/project/faiss-cpu/

Online technical sources were checked on 2026-07-27.
