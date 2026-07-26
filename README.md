# Applying Large Language Models to Develop a Retrieval-Augmented AI Tutor Prototype for Geotechnical Engineering

> **Status:** Work in progress.

## Project overview

This repository contains the implementation and experimental workflow for an individual master's dissertation in the MSc Data Science, AI and Digital Business programme at Gisma University of Applied Sciences.

The project develops and evaluates a retrieval-augmented generation (RAG) AI tutor prototype for undergraduate geotechnical engineering. It applies and extends the methodology presented by Tophel et al. (2025), while comparing newer large language model versions and testing controlled changes to the RAG configuration.

## Research questions

1. How effectively can the methodology described by Tophel et al. (2025) be applied to develop a working Retrieval-Augmented AI tutor prototype for geotechnical engineering?
2. How do newer LLM versions perform within this applied tutor setup compared to the versions evaluated in the baseline study?
3. How do small, controlled parameter changes, such as temperature, retrieval settings and chunk sizes, influence the tutor's accuracy and explanation quality within the same architecture?
4. How can some of the limitations identified in the baseline approach, such as formula reliability, limited multimodal capability or reasoning consistency, be addressed?

## Research design

The thesis follows an applied and development-based research design because its main technical contribution is a functional RAG-based tutor prototype. It also includes experimental and benchmarking elements through controlled comparisons of LLM versions and system parameters.

The planned pipeline uses:

- content extracted from *Principles of Geotechnical Engineering* by Das and Sobhan (2014);
- document chunking and retrieval;
- FAISS for vector similarity search;
- a LangChain-style RAG pipeline;
- selected LLM APIs for answer generation;
- controlled experiments in which the dataset, prompts and pipeline remain consistent when models are compared.

A sensitivity analysis will examine parameters such as temperature, retrieval depth and chunk size.

## Evaluation plan

The baseline study uses 391 textbook questions as the wider evaluation foundation and focuses its performance benchmark on 20 challenging questions. This project will preserve that distinction.

The focused evaluation will assess:

- answer accuracy;
- formula integration;
- clarity and utility of explanations;
- adaptability across different problem types.

The wider question set may be evaluated fully or through a documented representative subset, depending on feasibility and the available timeframe.

## Repository structure

| Path | Purpose |
|---|---|
| `configs/` | Experiment and pipeline configurations |
| `data/raw/` | Original authorised research inputs |
| `data/interim/` | Intermediate data-processing outputs |
| `data/processed/` | Final processed datasets used by experiments |
| `docs/` | Technical and research documentation |
| `notebooks/` | Exploratory analysis and experiment notebooks |
| `results/figures/` | Generated figures |
| `results/tables/` | Generated result tables |
| `results/metrics/` | Evaluation metrics |
| `src/` | Reusable implementation code |
| `tests/` | Automated tests and validation checks |

## Reproducibility

The project uses a dedicated Python virtual environment. `requirements.in` records the intentionally selected top-level dependencies, while `requirements-lock.txt` records the exact installed environment.

API keys and other secrets must remain in the local `.env` file and must never be committed. Copyrighted or restricted research sources will only be included when redistribution is permitted.
