# Retrieval-Depth Evaluation Protocol Amendment v3

## Status

This amendment was frozen on 21 September 2026 after the version 2
candidate-range adequacy gate was triggered and before any version 3 expanded
search, judgement migration, additional annotation or scoring was performed.

Version 3 is an adaptive extension of versions 1 and 2. It is not presented as
a decision that was specified before their results were observed.

## Reason for the amendment

Version 2 evaluated candidate depths 1, 2, 3, 4, 5, 8, 10, 15 and 20. Ranks
21-30 formed a diagnostic tail.

The frozen version 2 adequacy gate required retrieval-depth selection to stop
if a question without direct evidence in ranks 1-20 gained direct evidence in
ranks 21-30.

The gate was triggered by two questions:

- question `7.9` first gained direct evidence at rank 21;
- question `3.6a` first gained direct evidence at rank 28.

Direct evidence was available for 19 of the 20 questions by rank 30. Question
`6.6b` remained without direct evidence through rank 30.

Version 2 therefore correctly returned no selected retrieval depth. Although
depth 20 would have won among the version 2 candidates without the gate,
selecting it would have ignored direct evidence found in the diagnostic tail
and contradicted the frozen protocol.

## Alternatives considered

### Stop with no selected retrieval depth

The evaluation could stop after version 2 and report the depth as unresolved.
This would be methodologically valid, but it would leave the prototype without
a retrieval depth selected through the stated evaluation rule.

### Select depth 20 despite the gate

This option was rejected because it would override a decision rule after its
result was observed. It would weaken the traceability and credibility of the
evaluation.

### Add exact depths 21 and 28

This option was rejected because ranks 21 and 28 were directly observed from
the version 2 results. Using them as new candidate depths would tune the
candidate set too closely to individual evaluation outcomes.

### Expand directly beyond depth 30

A larger jump to depth 40 or 50 was rejected because the complete rank 21-30
range has already been judged and is sufficient to define the next candidate
boundary. A larger jump would increase annotation work and context size
without evidence that such a wide candidate range is yet required.

### Selected option

Version 3 will promote the full previous diagnostic range into the candidate
range and will retain a new ten-rank diagnostic tail. This continues the same
controlled expansion logic used between versions 1 and 2.

## Version 3 candidate range

Version 3 will use these candidate depths:

1, 2, 3, 4, 5, 8, 10, 15, 20, 25 and 30.

The maximum candidate depth will be 30.

The judgement pool depth will be expanded from 30 to 40. Ranks 31-40 will form
the new diagnostic tail and will not automatically become candidate depths.

Depths 25 and 30 are rounded operational depths. They were not selected by
copying the exact observed direct-evidence ranks of 21 and 28.

## Unchanged selection rule

The version 1 and version 2 selection rule will remain unchanged:

1. Retain candidate depths with the maximum direct-evidence hit count.
2. Among those depths, retain depths with the maximum useful-evidence hit
   count.
3. Select the smallest remaining depth.

MRR, graded nDCG, useful-evidence precision and context size will remain
diagnostic metrics. They will not become selection tie-breakers.

No retrieval depth will be selected until every required version 3 relevance
judgement is complete.

## Version 3 adequacy gate

The version 3 adequacy gate will trigger if a question without direct evidence
in ranks 1-30 gains direct evidence in ranks 31-40.

If the gate does not trigger, the unchanged frozen selection rule will be
applied.

If the gate triggers, no retrieval depth will be selected. Version 3 is the
final planned candidate-range expansion within this thesis. A further automatic
expansion will not be performed. The unresolved result will instead be
reported as evidence that the current embedding, chunking or dense-retrieval
configuration requires a separate methodological investigation.

This terminal boundary limits repeated adaptive tuning and keeps the evaluation
within the scope of the dissertation.

## Query-embedding reuse

The twenty evaluation questions, embedding model, embedding dimensions,
normalisation procedure and FAISS index remain unchanged.

Version 3 will reuse the exact query-embedding matrix already used by versions
1 and 2. No new embedding API request will be made.

The matrix must retain this SHA-256 fingerprint:

`fddd63335e250d875781890cec48cc646165e221d9b0cabdd22f489dc35be882`

The matrix must retain shape 20 by 1536 and pass the existing finite-value,
dtype, contiguity and normalisation checks.

## Existing-judgement reuse

The version 2 candidates at ranks 1-30 must be compared with the version 3
candidates at the same question and rank.

Reuse is permitted only if all 600 pairs match on the question identifier,
rank, chunk identifier, index position, text fingerprints, provenance and
similarity score.

After this equality check passes, the 600 completed version 2 grades and notes
may be transferred to new version 3 judgement records. The transfer must join
records through validated question-chunk pairs and must generate new judgement
identifiers from the version 3 configuration identifier.

Only the 200 pairs at ranks 31-40 will remain ungraded. They will be assessed
through the same blinded interface, grade definitions and positive-grade note
requirement used in versions 1 and 2.

The transfer procedure must never infer, alter or regenerate a human grade or
note.

## Leakage controls

Ground-truth answers will remain unavailable during retrieval and relevance
judgement.

The annotation interface will continue to hide retrieval rank, similarity
score, question and chunk identifiers, candidate-depth membership and source
provenance.

No model-generated answer or AI-generated relevance recommendation may be used
during the additional human judgments.

The evaluation questions will remain outside the corpus index.

## Output isolation

All version 1 and version 2 configurations, evidence and results will remain
unchanged.

Version 3 will use:

- a new configuration identifier;
- a separate configuration file;
- a separate lineage contract;
- separate preparation and audit outputs;
- a separate private relevance-judgement file;
- separate tracked metrics and metrics-table files;
- explicit refusal to overwrite version 1 or version 2 artifacts.

## Frozen version 2 lineage

The version 2 evidence is fixed by these Git commits:

- protocol amendment: `98f3e92`;
- configuration and lineage contracts: `228e21a`;
- preparation implementation and tests: `0b4a990`;
- preparation and migration audit: `e5f3f0c`;
- blinded annotation notebook: `8fbf7b8`;
- completed blinded judgements: `78d24f7`;
- retrieval-depth adequacy result: `3c32e86`.

The main version 2 fingerprints are:

- configuration:
  `04b69b8072509f165cf135478349d4e4a7030928c96036082258cd200b98358e`;
- lineage contract:
  `36226e0424594391ea9f2ee491aa72aa0d278ce00c288e95cc0b3e873bd0d6d3`;
- reused query-embedding matrix:
  `fddd63335e250d875781890cec48cc646165e221d9b0cabdd22f489dc35be882`;
- ranked candidate pool:
  `bebcb1564d946ee127b5d24be22a8b5e2cd170bcf83d7fd7f464c2e072fefeee`;
- pre-annotation judgement template:
  `614aaa969d7183c23f5a39f9b63da37fc4d2d8af6c6b746b1ad245f95cec4c33`;
- completed relevance judgements:
  `8b881829769b5c34d1064fed70fb389b0c811a0f1ef9d90fa96a769b6f03bf4a`;
- preparation summary:
  `f508a28cd5f019050c745ebaeac527f29a5a17ffb42ed360d39740d66cc930ad`;
- preparation and migration audit:
  `05f902a8b5068959444c960152fb02cd4e2358bb6a392d6630ac1939afff6ab7`;
- completed-judgement audit:
  `cc23157f20ec3a96ea9dfffe90a3983013b8b894afa8719655a97e8b1ac65b30`;
- metrics:
  `3dd489b55db418f170e7cd13393228303b7cdfc1a5764da90e8ea03650d0341b`;
- metrics table:
  `741f40f8a0bdc3b73c2d47304d4c176732e1e2acb4b9678cd3d23850dcae29e0`;
- independent scoring audit:
  `61f4c50eed11544626e6b843122a16ccbda61db5f4c9a57694ac9f9f43b66517`;
- final notebook:
  `c51ca0d253fc505ea20fe68855265c84278c0eb5044136919998787785993368`;
- FAISS index:
  `e049fa90324346c764c849071dbc49b7bc0cce125bc2fe8331a6f560fea8bdd1`;
- evaluation questions:
  `ec7c151ba709eb6e55d37290dc8fb349a5def8b204f439b8e354f71d0b97d599`.

## Adaptive-evaluation limitation

Versions 2 and 3 were designed after earlier adequacy-gate results were known.
They are adaptive protocol extensions and must be reported transparently.

The same 20 questions are used to tune the retrieval depth and later evaluate
the prototype. The selected depth, if resolved, is therefore benchmark-tuned
and is not an independent estimate of performance on unseen questions.

## Implementation boundary

This amendment records the version 3 methodological decision only.

No version 3 expanded search, judgement migration, additional annotation or
scoring may occur until:

1. the amendment is reviewed and committed;
2. the version 3 configuration and lineage contracts are committed;
3. the implementation supports the version 3 rank ranges;
4. automated tests pass;
5. the reuse and migration checks are independently audited;
6. every action flag is saved in its disabled state;
7. the implementation is committed before version 3 evidence is created.
