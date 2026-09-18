# Retrieval relevance-judgement guide

Status: frozen before the first production relevance judgement

Related configuration:
`configs/retrieval-evaluation-config.json`

## Purpose

This guide gives the operational interpretation of the three relevance
grades already frozen in the retrieval-evaluation configuration. It does
not change the protocol, candidate depths, metrics or selection rule.

The interpretation is recorded before the assessor views or grades the
first production question-chunk pair. This prevents later retrieval
results from changing how relevance is judged.

## Evidence boundary

Each judgement concerns one question and one retrieved corpus chunk.

The assessor may use technical knowledge to understand the displayed
text but must judge only the evidence actually present in the chunk.
Missing equations, assumptions or reasoning must not be supplied from
memory.

The assessor must not consult:

- retrieval rank or similarity score;
- candidate-depth membership;
- other retrieved chunks while judging the current pair;
- solution-manual or ground-truth answers;
- model-generated answers;
- external references or textbook pages outside the displayed chunk.

## Decision order

Apply the following questions in order:

1. Does the chunk provide the central concept, equation, method, data or
   worked procedure needed for the requested task or a substantial
   required part of it? If yes, assign grade 2.
2. If not, does it provide a useful definition, assumption,
   relationship, parameter explanation or auxiliary step that would
   materially support an answer? If yes, assign grade 1.
3. Otherwise, assign grade 0.

## Grade 0: not relevant

Assign grade 0 when the chunk:

- does not help construct the answer;
- only shares terminology with the question;
- discusses a different topic or method;
- contains an isolated fragment with no usable meaning;
- is too incomplete or corrupted to provide useful evidence;
- repeats the question without providing explanatory evidence.

No note is required for grade 0.

## Grade 1: supporting evidence

Assign grade 1 when the chunk provides useful evidence but not the main
evidence required to answer the question.

Examples include:

- a relevant definition when the question primarily requires a
  calculation;
- an assumption or condition needed before applying a method;
- an explanation of one parameter in the required equation;
- a useful relationship that still leaves the central procedure
  missing;
- a partial but interpretable equation fragment;
- background theory that helps explain the final answer.

A short note must identify the supporting evidence. It may also state
which central evidence is absent.

## Grade 2: direct evidence

Assign grade 2 when the chunk supplies the main evidence needed for the
requested task or a substantial required component that could not
reasonably be omitted.

Examples include:

- the central equation required for the calculation;
- the method or ordered procedure needed to solve the problem;
- a worked example that clearly demonstrates the required method, even
  when its numerical values differ;
- the main concept needed for an explanatory question;
- a usable table containing the principal values or relationships
  required by the question.

The chunk does not need to calculate the final numerical answer itself.
However, topic similarity or general background alone is insufficient
for grade 2.

A short note must identify the direct evidence provided.

## Edge cases

### Multi-part questions

A chunk may receive grade 2 when it directly addresses a substantial
required part of a multi-part question. Evidence for only a minor or
auxiliary part receives grade 1.

### Definitions

A definition normally receives grade 1 when the question requires a
calculation or broader procedure. It receives grade 2 when defining or
explaining the concept is the central task.

### Tables

A linearised table receives grade 2 when it contains the main data or
relationship required by the question. A partial table that is useful
but insufficient receives grade 1.

### Formula and extraction damage

Judge the usable evidence that remains. An unreadable central formula
does not receive grade 2. A damaged fragment may receive grade 1 only
when it still provides meaningful support.

### Repeated evidence

Judge every pair independently. Do not reduce a grade because similar
evidence appeared in an earlier pair.

## Judgement notes

Grades 1 and 2 require a short evidence-focused note.

Suitable patterns are:

- `Defines or explains [supporting concept], but does not provide
  [main missing method or equation].`
- `Provides [central equation, method, concept or data] needed for
  [the requested task].`

Avoid vague notes such as `relevant`, `good chunk` or `seems useful`.

Grade-0 records retain a null note.

## Consistency rule

The same evidence standard must be applied to all 400 pairs. Grades must
not be adjusted based on how many relevant chunks have already appeared
for a question or on assumptions about retrieval rank.

Ground-truth answers remain unavailable throughout annotation.
Retrieval depth remains unresolved until every judgement is complete
and the frozen scoring rule is applied.
