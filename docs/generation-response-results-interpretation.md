# Generation-response results: working interpretation

We report these results from the 140 fixed expert judgments and the public aggregate files committed in `9abcd8f`. The professor's current feedback on the overall system remains preliminary and can be discussed separately when it is received. It does not change these answer scores.

## Answer correctness and completion

| Condition | Model and role | Correct / 20 | Accuracy | Wilson 95% interval | Visible / 20 |
| --- | --- | ---: | ---: | ---: | ---: |
| P0 | `gpt-4-0613`, no retrieval | 2 | 10% | 2.8–30.1% | 20 |
| P1 | `gpt-4-0613`, retrieval, temperature 0.1 | 3 | 15% | 5.2–36.0% | 20 |
| P2 | `gpt-4-0613`, retrieval, temperature 0.5 | 3 | 15% | 5.2–36.0% | 20 |
| P3 | `gpt-4-0613`, retrieval, temperature 1.0 | 2 | 10% | 2.8–30.1% | 20 |
| M1 | `gpt-4.1-2025-04-14`, retrieval | 10 | 50% | 29.9–70.1% | 20 |
| M2 | `gpt-5-2025-08-07`, retrieval | 11 | 55% | 34.2–74.2% | 11 |
| M3 | `gpt-6-astra`, retrieval | 20 | 100% | 83.9–100% | 20 |

We use the same 20 prevalidated questions in every condition. M2 produced 11 visible answers and all 11 were judged correct, but the other nine returned no visible text. We therefore keep its primary accuracy at 11/20 and report its completion at 11/20. We do not describe 11/11 among visible answers as its overall benchmark accuracy. M3 answered all 20 correctly in this recorded run. Its `gpt-6-astra` identifier was a mutable alias, so the result is linked to the response-file fingerprint and may not describe a later version of that alias.

## Planned paired comparisons

P0 to P1 changed one question from incorrect to correct and none in the opposite direction. Exact McNemar gives *p* = 1.000. For the temperature conditions P1/P2/P3, Cochran's Q gives *Q*(2) = 0.667 and *p* = 0.717; the Holm-adjusted pairwise McNemar results are all *p* = 1.000. In this 20-question run we do not find evidence of a retrieval or temperature difference. These results do not establish that retrieval or temperature could never matter on other questions or configurations.

For P1/M1/M2/M3, Cochran's Q gives *Q*(3) = 31.286 and *p* = 7.40 × 10⁻⁷. Holm-adjusted exact McNemar comparisons with P1 give *p* = 0.03125 for M1, 0.02344 for M2, and 0.0000916 for M3. M3 also differs from M1 (*p* = 0.00977) and M2 (*p* = 0.01563), while M1 and M2 do not differ under this test (*p* = 1.000). We call these configuration comparisons: M2 and M3 also use reasoning settings and API parameter support that differ from P1. We cannot isolate the effect of the model name alone from those differences.

## Secondary criteria and errors

The formula, calculation and unit criteria use 17, 20 and 18 applicable questions respectively; clarity and task adaptability each use 20. M3 has a mean score of 2.00 for all five criteria in this run. M2's nine no-visible-answer records receive zero where applicable, which lowers its overall means despite the scores of its eleven visible answers. We report the five criteria separately from binary correctness.

The model-generation Friedman tests are below 0.05 for each secondary criterion; the temperature Friedman tests are not. Individual Wilcoxon results are interpreted using the Holm-adjusted values within each criterion's planned family. We did not adjust across all five secondary criteria, so we treat them as supporting and exploratory findings rather than five independent confirmations of the primary outcome.

Grounding and conceptual errors make up most of the P0–P3 incorrect answers. All nine M2 errors are in the explicit `no_visible_answer` category, and M3 has no incorrect answers in this set. We retain deficiency and other as documented additions to the three error categories described by Tophel et al. (2025). The separate unsupported-claims field is reported as its own count; it is not another mutually exclusive error category.

## Interpretation limits

The 20 questions are a fixed challenging benchmark and one output was generated for each question-condition pair. Wilson intervals summarise the accuracy proportions under a binomial model; the purposively selected questions do not justify treating the intervals as proof of general performance, and they do not measure repeated-generation variability. One expert scored all answers, and seeing seven blinded answers for a question together may have influenced judgments by comparison.

Ten frozen references have no recorded problem-type or cognitive-level classification. The ten classified references all carry the `Apply` cognitive label; their problem types are pooled in the public output because individual types have too few questions. The two published subgroup tables therefore describe the same ten-versus-ten split and do not provide separate evidence about each type of problem.

Tophel et al. (2025) report percentages such as 82.5% for their GPT-4 API setting. Their exact grading calculation cannot be recovered as our binary correct/20 measure, and their model snapshot and conditions are not identical to ours. We report the original percentages as context, not as directly comparable estimates or a test against our results. The later thesis discussion can add the professor's overall-system comments while preserving these fixed scores and the distinctions above.
