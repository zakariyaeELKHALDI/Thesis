# Retrieval-Depth Evaluation Protocol Amendment v2

## Status

This amendment was frozen on 20 September 2026 after the version 1
candidate-range adequacy gate was triggered and before any expanded search,
new relevance judgement or version 2 scoring was performed.

This is an adaptive extension of the version 1 protocol. It is not presented
as a decision that was specified before the version 1 results were observed.

## Reason for the amendment

Version 1 evaluated candidate depths 1, 2, 3, 4, 5, 8 and 10, while ranks
11-20 were retained as a diagnostic tail.

The frozen adequacy gate required retrieval-depth selection to stop if a
question without direct evidence in ranks 1-10 gained direct evidence in
ranks 11-20.

The gate was triggered for five of the twenty evaluation questions.
Consequently, version 1 correctly produced no selected retrieval depth.
Selecting only among depths up to 10 would have ignored direct evidence
already observed in the diagnostic tail.

## Version 2 candidate range

Version 2 will use the following candidate depths:

1, 2, 3, 4, 5, 8, 10, 15 and 20.

The maximum candidate depth will therefore be 20.

The judgement pool depth will be expanded from 20 to 30. Ranks 21-30 will
form a new diagnostic tail and will not automatically become candidate
depths.

This design promotes the complete former diagnostic range into the candidate
range while preserving a separate ten-rank diagnostic tail. Depths 15 and 20
are rounded operational depths and were not selected by copying the exact
ranks at which direct evidence appeared in version 1.

## Unchanged selection rule

The version 1 selection rule will remain unchanged:

1. Retain candidate depths with the maximum direct-evidence hit count.
2. Among those depths, retain depths with the maximum useful-evidence hit
   count.
3. Select the smallest remaining depth.

MRR, graded nDCG, useful-evidence precision and context size will remain
reported diagnostic metrics and will not become selection tie-breakers.

No retrieval depth will be selected until every required version 2
relevance judgement is complete.

## Version 2 adequacy gate

The version 2 adequacy gate will trigger if a question without direct
evidence in ranks 1-20 gains direct evidence in ranks 21-30.

If the gate triggers, no retrieval depth will be selected. A further
versioned protocol decision will be required before another expansion.

If the gate does not trigger, the unchanged frozen selection rule will be
applied.

## Query-embedding reuse

The twenty evaluation questions, embedding model, embedding dimensions,
normalisation procedure and FAISS index are unchanged.

Version 2 will therefore reuse the exact version 1 query-embedding matrix.
No new embedding API request will be made.

The reused matrix must retain this SHA-256 fingerprint:

`fddd63335e250d875781890cec48cc646165e221d9b0cabdd22f489dc35be882`

The matrix must also retain shape 20 by 1536 and pass the existing finite
value and normalisation checks.

Reusing the frozen matrix avoids unnecessary cost and removes possible
variation from a second API response.

## Existing-judgement reuse

The version 1 ranked candidates at ranks 1-20 must be compared with the
version 2 candidates at the same question and rank.

Reuse is permitted only if all 400 pairs match on the question identifier,
rank, chunk identifier, index position, text fingerprints, provenance and
similarity score.

After this equality check passes, the 400 completed version 1 grades and
notes may be transferred to new version 2 judgement records. The transfer
must join records through validated question-chunk pairs and must generate
the new judgement identifiers from the version 2 configuration identifier.

Only the 200 pairs at ranks 21-30 will remain ungraded. They will be assessed
using the same blinded interface, grade definitions and positive-grade note
requirement used in version 1.

The transfer procedure must never infer, alter or regenerate a human grade.

## Leakage controls

Ground-truth answers will remain unavailable during retrieval and relevance
judgement.

The annotation interface will continue to hide retrieval rank, similarity
score and candidate-depth membership.

No model-generated answer or AI-generated relevance recommendation may be
used during the additional human judgements.

The evaluation questions will remain outside the corpus index.

## Output isolation

The version 1 configuration and every version 1 artifact will remain
unchanged.

Version 2 will use:

- a new configuration identifier;
- a separate configuration file;
- separate preparation and audit outputs;
- a separate private relevance-judgement file;
- separate tracked metrics and metrics-table files;
- explicit refusal to overwrite version 1 artifacts.

## Frozen version 1 lineage

The version 1 evidence is fixed by the following Git commits:

- completed blinded judgements: `20026e1`;
- initial retrieval-depth adequacy result: `e71c06a`.

The relevant version 1 fingerprints are:

- configuration:
  `c99ee459c389407dd77e710b8b2ec5081d51889d0d1e5a713d5dcdf9d5d68c0f`;
- query-embedding matrix:
  `fddd63335e250d875781890cec48cc646165e221d9b0cabdd22f489dc35be882`;
- ranked candidate pool:
  `1713a14a25433abde59791eede325f33605da86da85019e2cda3597ec0ffc114`;
- empty pre-annotation judgement template:
  `bbc72d5e423bf5d745ec93cb9b6de952520135039a3f6dac5132b3a784b718e1`;
- completed relevance judgements:
  `33f408a2f744d662f78d9eaa6c77d9a1f321d487899f7a2758bf162646ca8fe9`;
- preparation summary:
  `6083b0048d45905cd6724b472a6ec6f577a16f5c98984e9bde4fa8b01402e040`;
- initial metrics:
  `1b60b2b7accc7cb75716adea5e0dc90936d1cbddb08d9c0aec03f4a7c6ace453`;
- initial metrics table:
  `f152ef7b5b41fb9defedd8630da4f28eb4673457940742ad79f19331db9292f5`;
- FAISS index:
  `e049fa90324346c764c849071dbc49b7bc0cce125bc2fe8331a6f560fea8bdd1`;
- evaluation questions:
  `ec7c151ba709eb6e55d37290dc8fb349a5def8b204f439b8e354f71d0b97d599`.

## Implementation boundary

This amendment records the version 2 methodological decision only.

No expanded search, judgement migration, additional annotation or version 2
scoring may occur until:

1. the version 2 configuration and implementation are complete;
2. automated tests pass;
3. the reuse and migration integrity checks are independently audited;
4. all action flags are saved in their disabled state;
5. the implementation is committed before creating version 2 evidence.
