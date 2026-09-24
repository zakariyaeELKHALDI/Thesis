# Generation-response analysis method

## Frozen input and current status

We analyse the 140 blinded judgments imported after the grouped review. Their exact SHA-256 is `ad861d0d40bc4bfa59a0bda5b6f57abbff0aaac4b772cb9daad7adc4317ca567`; the completed private scoring notebook SHA-256 is `56f87d4b53d60a6dbfd2447299237259ec5492f7cbff115e993fe7ff2da39bce`. The private checkpoint, the preliminary-confirmation status record and a separate decision that these scores are final for analysis are all checked before we attach conditions in memory. We do not rewrite the imported scores. The professor's further comment on the overall system belongs in the later interpretation of results.

## Aggregate outcomes

We calculate binary accuracy separately for all seven conditions from the same 20 prevalidated questions. Wilson 95% confidence intervals describe each 20-question accuracy. We report visible-answer completion separately and keep all nine responses without visible text in the accuracy denominator as incorrect.

Formula integration, calculation correctness and unit correctness are summarised only for questions marked applicable in the approved reference. We also report explanation clarity, task adaptability and primary error categories. Question-level data, private notes, generated responses and reference text never enter public tables.

## Paired comparisons

We use exact McNemar for P0 versus P1. P1/P2/P3 and P1/M1/M2/M3 use Cochran's Q and exact pairwise McNemar comparisons. We adjust pairwise p-values with Holm's method within each preplanned comparison family. Secondary ordinal scores use paired Wilcoxon tests for two conditions and Friedman tests followed by paired Wilcoxon tests for three or four conditions, restricted to the same prevalidated applicable questions across that family. The ordinal tests account for ties and report the common applicable denominator. No claim of within-condition generation variance follows from one response per question-condition pair.

The original paper's GPT-4 versus Llama-3 API pair is absent from our frozen conditions. We therefore do not represent an unpaired chi-square test for that original pair as a replication. Our primary comparisons follow the paired 20-question design. The article's 82.5% result cannot be reconstructed as a single binary correct/20 measure without unpublished grading details.

## Descriptive strata and limitations

We summarise task adaptability across approved `problem_type` and `cognitive_level` values. Ten approved references have neither classification recorded; the analysis reports them as `not_recorded_in_reference`, without inferring labels from generated responses. When any recorded category for a field contains fewer than three questions, we pool all recorded categories for that field as `recorded_not_stratified` before publishing counts and scores. All seven recorded problem types require pooling; the ten `Apply` cognitive-level records can be reported together. These strata are descriptive and do not support individual problem-type comparisons. The thesis also reports that seven answers were visible together within each question, one domain expert provided the judgments, and the professor's feedback on the overall system is separate from the fixed answer scores.

The public notebook and module contain code and aggregate-only displays. The results export creates condition accuracy, secondary score, paired comparison and error-distribution tables plus one metrics JSON. These files contain no question or answer text, response identifier, question identifier or judgment note. Publication refuses to replace existing outputs silently.
