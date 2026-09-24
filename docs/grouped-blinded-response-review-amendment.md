# Grouped blinded response review amendment

## Timing and scope

We changed the presentation and capture of expert judgments on 24 September 2026, after committing the frozen reference package in `6b80434` and before scoring any generated answer. The 20 approved reference records, three discrepancy decisions, 140 generated responses and frozen rubric remain unchanged. This amendment replaces the presentation and checkpointing procedure in the earlier response evaluation protocol.

## Expert review file

Before scoring began, we chose a private, Git-ignored Jupyter notebook as the editable review file instead of the initially planned Word form. Notebook 07 remains the public reference validation record; the offline tool in `src/geotech_rag/grouped_response_review.py` generates the private scoring notebook from the frozen inputs. We use 20 sections in the frozen question order. Each section contains the benchmark question, validated reference solution, approved grading requirements and seven generated answers. The notebook also presents the four evaluation criteria and four reported error types from Tophel et al. (2025), followed by our frozen scoring scales, operational error definitions and the two documented additional error categories. We label the answers A to G after a deterministic shuffle within each question. The scoring notebook contains no condition, model, retrieval setting, temperature, API metadata or source response identifier. We retain the label-to-response mapping in a separate Git-ignored private file.

The expert and researcher record seven separate scores per question in editable code cells. They save the notebook without executing those cells and confirm the completed assessment with Professor Peña Olarte. Nine responses with no visible answer remain in their assigned question groups. We show them as having no visible answer and apply the frozen empty-response rule without exclusion or regeneration.

Viewing all seven answers to a question together permits direct comparison between them. This may influence individual scores. We will report grouped presentation as a methodological limitation when interpreting condition-level results. Question eligibility, accepted methods and numerical tolerances remain fixed independently of the generated answers.

## Validation and import

The original blank notebook remains private and unchanged. We keep the completed copy separately. The offline validator reads scoring cells as data without executing them. It checks the response and reference fingerprints, the 20-by-seven structure, the private mapping, allowed values, applicability, required notes and the nine empty-response decisions. It refuses missing, duplicate or inconsistent entries. Only a complete set of 140 valid judgments is imported atomically to the Git-ignored judgment file after explicit confirmation that the professor has reviewed the scores. The import records blinded response identifiers and scoring fields without question text, answer text, model or condition. Completed judgments cannot be silently replaced; any later correction requires an explicit private audit trail under the frozen change-control policy.

We use the offline tool's `generate` operation to create the untouched notebook and its separate label mapping. The `start` operation creates a distinct editable copy. We save that completed notebook after scoring and use `validate` to check it without executing any notebook cell or writing a judgment. Following the professor's confirmation, `import --expert-confirmed` writes the validated judgment set and private fingerprint metadata once.

Unblinding and condition-level analysis remain in notebook 08 after all judgments pass validation. The original binary correctness outcome, secondary scales, error categories, paired design and planned statistical tests are unchanged.
