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
