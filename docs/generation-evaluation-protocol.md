# Generation Evaluation Protocol

## Status

This protocol was frozen on 22 September 2026 after retrieval depth 30
had been selected and the production retrieval implementation had passed
its independent audit.

It was frozen before any benchmark answer-generation request was made.

## Purpose

The experiment approximates the GPT-4 experiments reported by Tophel et
al. (2025) and compares that historical baseline with newer OpenAI model
generations.

The design has two distinct parts:

1. a paper-baseline approximation using GPT-4 under every documented
   temperature setting;
2. a controlled temporal comparison using the same retrieval evidence
   and benchmark questions with newer models.

The experiment is an approximation rather than an exact replication
because the paper does not identify a fixed GPT-4 snapshot, exact
zero-shot prompt, output-token limit, retry policy, or complete response
scoring procedure.

## Frozen benchmark inputs

All conditions use the same twenty text-only benchmark questions.

For retrieval-augmented conditions, the following production retrieval
contract is fixed:

- embedding model: text-embedding-ada-002;
- vector index: FAISS IndexFlatIP;
- selected retrieval depth: 30;
- retrieved chunks: the exact frozen production top-30 results;
- retrieved chunk order: descending similarity order;
- query embeddings: the frozen benchmark query matrix;
- new embedding API requests during benchmark generation: prohibited.

No ground-truth answer or relevance grade may be included in an API
request.

## Experimental conditions

The experiment contains seven conditions and 140 generated responses.

| ID | Model | Retrieval | Temperature | top_p | Reasoning |
|---|---|---:|---:|---:|---|
| P0 | gpt-4-0613 | none | 0.1 | 1 | not applicable |
| P1 | gpt-4-0613 | k=30 | 0.1 | 1 | not applicable |
| P2 | gpt-4-0613 | k=30 | 0.5 | 1 | not applicable |
| P3 | gpt-4-0613 | k=30 | 1.0 | 1 | not applicable |
| M1 | gpt-4.1-2025-04-14 | k=30 | 0.1 | 1 | not applicable |
| M2 | gpt-5-2025-08-07 | k=30 | omitted | omitted | low |
| M3 | gpt-6-astra | k=30 | omitted | omitted | low |

Each condition produces exactly one response for each benchmark question.

No response may be regenerated merely because it appears incorrect or
unsatisfactory.

## Paper-baseline approximation

Conditions P0 to P3 form the paper-baseline approximation.

The paper evaluated GPT-4 with RAG at temperatures 0.1, 0.5 and 1.0,
while keeping top_p at 1. These three settings are reproduced in P1,
P2 and P3.

P0 is a controlled zero-context condition at temperature 0.1. The same
tutor instructions and question wording are used, but retrieved context
is omitted.

The paper presents only one GPT-4 zero-shot result and does not document
enough information to recreate its exact prompt or generation settings.
P0 must therefore be described as a defined zero-context control, not an
exact reproduction of the paper's zero-shot experiment.

## Newer-model comparison

M1 uses the fixed GPT-4.1 snapshot and the paper's best-performing
temperature setting.

M2 uses the fixed GPT-5 snapshot. Its reasoning effort is fixed at low.
Temperature and top_p are omitted because reasoning-model controls are
not directly equivalent to the GPT-4 sampling parameters.

M3 uses GPT-6 Astra at low reasoning effort. No dated Astra snapshot was
available when this protocol was frozen. It is therefore an exploratory,
time-bound comparison.

For M3, the generation record must retain:

- requested model identifier;
- returned model identifier;
- response identifier;
- request and response timestamps;
- reasoning setting;
- token usage;
- service fingerprint or equivalent metadata when supplied.

## Model availability preflight

The OpenAI model-list endpoint was queried on 22 September 2026 using
OpenAI Python SDK version 2.53.0.

The following required models were listed as available:

- gpt-4-0613;
- gpt-4.1-2025-04-14;
- gpt-5-2025-08-07;
- gpt-6-astra.

The model-list request transmitted no benchmark question, retrieved
chunk, answer, or ground-truth text and incurred no generation request.

If a required model becomes unavailable before generation, execution
must stop. Substitution requires a versioned protocol amendment.

## Llama-3 boundary

The paper also compared GPT-4 with a Llama-3 model described as having
70 billion parameters.

Llama-3 is not part of the present OpenAI experiment because the paper
does not freeze the exact model build, provider, quantisation, serving
framework, hardware configuration, or decoding implementation.

A Llama-3 experiment may be added only through a separate versioned
protocol that freezes those details before generation. Its exclusion
does not remove any GPT-4 parameter setting reported by the paper.

## Prompt controls

Every RAG condition uses the same frozen tutor prompt.

The prompt adapts the ten tutor characteristics from the paper:

1. personalisation;
2. interactive and engaging explanation;
3. accessibility;
4. feedback and support;
5. motivation;
6. resourcefulness;
7. flexible learning paths;
8. contextual understanding;
9. assessment and evaluation;
10. safety and privacy.

The adapted prompt must also instruct the model to:

- use the supplied textbook evidence;
- distinguish supplied information from assumptions;
- show equations, substitutions, units and intermediate calculations;
- state when the supplied evidence is insufficient;
- avoid inventing textbook facts, equations or references;
- answer the current question without relying on conversation history.

The exact prompt text and its SHA-256 fingerprint must be committed
before generation.

## Fixed request controls

The following request controls apply:

- one independent request per question and condition;
- no conversation history;
- no tools, web search or code execution;
- streaming disabled;
- one returned answer;
- maximum output length fixed identically across conditions;
- no seed unless supported identically by every selected model;
- no silent model substitution;
- no answer regeneration following a successful response;
- transient failures retried only under the frozen retry policy.

The endpoint, maximum output length, timeout policy and retry policy
must be fixed in the committed generation configuration before any paid
request.

## Output and checkpoint controls

Every response record must contain:

- configuration and condition identifiers;
- blinded response identifier;
- question identifier;
- retrieved-evidence fingerprint;
- prompt fingerprint;
- requested and returned model identifiers;
- sampling or reasoning parameters;
- response identifier;
- request and response timestamps;
- token usage;
- response text;
- retry count;
- completion status.

Private response records containing question, context or answer text
must remain Git-ignored.

Generation must be resumable. Completed successful records must not be
overwritten during a resumed run.

Tracked artifacts may contain aggregate metrics, identifiers,
fingerprints and text-free audit evidence only.

## Evaluation protocol

Generation must finish before any ground-truth answer is accessed for
response evaluation.

Responses will be placed in a deterministic blinded order that hides:

- model;
- condition;
- temperature;
- reasoning effort;
- retrieval status;
- response metadata.

The primary outcome is binary answer correctness for each question.

Secondary evaluation fields will cover:

- formula selection and integration;
- calculation correctness;
- unit correctness;
- clarity and usefulness of the explanation;
- adaptability to the problem type;
- unsupported or hallucinated claims;
- conceptual, calculation, grounding and deficiency error categories.

Wilson 95 percent confidence intervals will be reported for binary
accuracy.

Because every condition answers the same questions, paired statistical
tests will be used for the main comparisons. A paper-comparable
chi-square calculation may be reported secondarily, but it will not
replace the paired analysis.

The paper reports 82.5 percent accuracy for one twenty-question
condition, which cannot be produced by twenty binary scores alone.
Because its scoring denominator is not documented, this experiment will
not attempt to reverse-engineer that percentage. Its explicit binary
rubric and denominator will be reported instead.

## Stochasticity boundary

Only one response per question and condition will be generated.

This follows the apparent paper design and keeps human evaluation
feasible, but it does not estimate within-condition generation
variance. This limitation must be reported when interpreting the
temperature results.

## Execution boundary

No paid generation may occur until:

1. this protocol is committed;
2. the exact prompt and generation configuration are committed;
3. the offline context-window and token-budget audit passes;
4. the generation implementation and automated tests pass;
5. the private-output and checkpoint controls are independently audited;
6. every action flag is saved in its disabled state;
7. a maximum expected cost is calculated and approved;
8. the implementation is committed before creating generation evidence.
