# Generation Response Evaluation Protocol

## Purpose

This protocol defines the post-generation evaluation of the
140 frozen responses produced from 20 benchmark questions and
seven experimental conditions. It is frozen before reference
solutions are reconstructed and before any response receives a
correctness judgment.

The protocol supports the thesis as applied,
development-based, experimental and benchmarking research. It
provides explicit KPIs, controlled comparisons and statistical
procedures, in line with the dissertation assessment
requirements for justified methodology and thorough
interpretation of results.

## Relationship to the baseline study

Tophel et al. evaluate accuracy, formula integration,
explanation clarity and adaptability. The paper also reports
grounding, conceptual, calculation and deficiency errors.

However, the paper does not publish a reproducible
response-level rubric. It does not explain how partial answers,
equivalent methods, rounding, units, multi-part answers or
incomplete solutions were judged. Its reported accuracy of
82.5% cannot be produced by one binary judgment across 20
questions because binary accuracy would change in increments of
five percentage points.

This study therefore retains binary question-level accuracy as
the primary baseline-compatible measure, while explicitly
operationalising the missing judgment rules. These additions
are documented as methodological clarifications rather than
claims that the baseline paper used the same detailed rubric.

## Evaluation order

The evaluation must follow this order:

1. Freeze and commit this public protocol and its JSON
   configuration.
2. Reconstruct the 20 private reference solutions from the
   authorised solution manual.
3. Have the geotechnical engineering expert validate all
   reference solutions without access to generated responses.
4. Freeze and fingerprint the validated reference-answer file.
5. Present all 140 responses to the expert in deterministic
   blinded order.
6. Validate the completed blinded judgment set.
7. Unblind the scores and conduct the statistical analysis.

Generated answers may not influence question eligibility,
reference-answer content, accepted methods or numerical
tolerances.

## Reference-solution reconstruction

The solution manual will be used to reconstruct one reference
record for each benchmark question. The reconstructed solution
will be written in readable Markdown, with LaTeX used for
equations, variables, subscripts, superscripts and mathematical
symbols.

Automatic PDF extraction may assist navigation, but equations
and symbols must be manually checked against the source pages.
The reference record will preserve source-page provenance,
expected results, accepted equivalent methods, required
formulae, units and question-specific tolerances.

Prof. Peña Olarte will validate all 20 reference records before
the generated response file is loaded for scoring.

## Primary outcome: binary answer correctness

The primary outcome is `answer_correctness`, scored as either
1 or 0.

A score of 1 requires all materially requested results, a
technically valid method and no material engineering error.
Equivalent valid methods are accepted.

A score of 0 is assigned when:

- a requested final result is missing;
- a material part of a multi-part answer is missing or wrong;
- a numerical result falls outside the frozen tolerance;
- materially incorrect reasoning produces an apparently
  correct result;
- required units are wrong or make the answer
  uninterpretable;
- a material unsupported claim changes the meaning; or
- no visible usable answer is returned.

This means a response that describes the correct method but
asks the user to finish the calculation is incorrect for the
primary outcome. Its useful method can still receive partial
secondary scores.

## Secondary judgments

The expert also records:

- formula integration: not applicable, 0, 1 or 2;
- calculation correctness: not applicable, 0, 1 or 2;
- unit correctness: not applicable, 0, 1 or 2;
- explanation clarity: 0, 1 or 2;
- task adaptability: 0, 1 or 2;
- presence of unsupported claims;
- one primary error category;
- judgment confidence; and
- a private explanatory note where required.

Applicability is frozen in the validated reference record,
rather than decided differently for individual model answers.

## Error categories

The baseline categories are retained:

- grounding;
- conceptual;
- calculation; and
- deficiency.

Two documented operational categories are added:

- `no_visible_answer`, for a successful API record without
  visible answer text; and
- `other`, for an error that cannot be represented faithfully
  by the preceding categories.

Correct responses use the category `none`. Incorrect responses
must use a non-`none` category.

## Empty responses

The nine frozen M2 responses without visible answer text remain
part of the experiment. They are not regenerated, removed or
replaced.

They receive binary correctness 0, explanation clarity 0, task
adaptability 0 and the primary error category
`no_visible_answer`. Formula, calculation and unit scores are
either not applicable or 0 according to the corresponding
prevalidated question requirements.

The domain expert confirms each record through the blinded
interface. Completion rate is reported separately from
correctness.

## Expert evaluation

Prof. Peña Olarte will review all 140 responses. No sample will
be used to extrapolate judgments to the remaining responses.

The evaluation uses one domain expert. Consequently, the thesis
will not claim inter-rater reliability and will report the
single-expert design as a limitation.

## Blinding

The expert sees only:

- the benchmark question;
- the validated reference solution;
- one candidate response; and
- the frozen rubric.

Model identity, condition, retrieval status, temperature,
reasoning effort, token usage, latency and API metadata remain
hidden. Responses are presented in deterministic shuffled order
and answers to the same question are not deliberately grouped.

The blinded notebook cannot reveal condition-level results.

## Checkpointing and validation

Each judgment is saved atomically. Evaluation can resume after
interruption without regenerating or silently overwriting
completed records.

The private judgment file stores the blinded response ID and
evaluation fields, but does not duplicate question text,
response text, model identity or condition identity.

Unblinding is blocked until 140 unique, structurally valid
judgments are present and all reference and response
fingerprints still match their frozen values.

## Statistical analysis

Accuracy is reported per condition with Wilson 95% confidence
intervals.

Because the same 20 questions are answered under every
condition, paired tests are primary:

- P0 versus P1: exact McNemar test;
- P1, P2 and P3: Cochran's Q followed by exact pairwise
  McNemar tests;
- P1, M1, M2 and M3: Cochran's Q followed by exact pairwise
  McNemar tests.

Pairwise p-values are adjusted using the Holm method.

Applicable ordinal secondary scores use paired Wilcoxon tests
for two conditions and Friedman tests followed by paired
Wilcoxon tests for larger groups. The common applicable
denominator is always reported.

A paper-style unpaired chi-square comparison may be included
only as a secondary compatibility analysis. It does not replace
the paired primary analysis.

Adaptability is interpreted using the response-level score and
performance across the prevalidated problem-type and
cognitive-level strata. Small strata are treated
descriptively.

No within-condition stochastic variance is claimed because
only one frozen response exists for each question-condition
pair.

## Notebook separation

`07_blinded_response_evaluation.ipynb` performs reference
validation and blinded expert scoring. It cannot unblind
results or calculate condition-level performance.

`08_response_evaluation_analysis.ipynb` performs unblinding,
statistical comparisons, tables, charts and research-question
interpretation only after all 140 judgments pass validation.

This separation reduces accidental unblinding and keeps the
manual evaluation independent from the final comparisons.

## Privacy and tracked outputs

Questions, reference answers, generated responses, evaluator
notes and per-response judgments remain private and
Git-ignored.

Public outputs contain only text-free aggregate metrics,
condition-level summaries, statistical comparisons, figures
and documented fingerprints.

## Change control

Any change before scoring must be documented, tested and
committed before evaluation begins.

If a scoring rule must change after scoring starts, evaluation
stops. The amendment must be documented, affected judgments
invalidated and all affected responses rescored under one
consistent protocol. Silent rule changes are prohibited.
