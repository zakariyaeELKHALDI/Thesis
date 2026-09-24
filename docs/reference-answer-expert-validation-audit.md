# Reference-answer expert validation audit

This audit records the frozen reference validation step in our generation-response evaluation. It contains no private benchmark question, reference answer, reviewer note or generated response.

## Expert decision and scope

- We recorded approval of the 20 reference answers, their acceptance rules and all three source-discrepancy decisions on 2026-09-24. The private metadata stores the approval source.
- The review form SHA-256 is `0faaccf8a0d79dfb0426c3a3b3b2ed73ca68b4e57e9f0d25096deea4488da1cf`.
- We did not open generated response texts during reference validation.

## Private-file fingerprints

| Private file | Before approval SHA-256 | Approved/frozen SHA-256 |
| --- | --- | --- |
| `benchmark-reference-answers.jsonl` | `e51162fdde2884a8e6f59219c9d4f68db4e01a791d524db7520695984b37e6d2` | `981650c4f1cdf622bffaec0f84f263105b5ff1b2f6a41ecb273577c9410cb5c8` |
| `benchmark-reference-source-discrepancies.jsonl` | `55babe7c64ad94a84e2fcff701a23d0bc2eab113d76ec9cfe88a14edf53a1613` | `7ba6ccc135f5354c85fff60b60781b5700691abe8f76477779f72ffef73e8571` |
| `benchmark-reference-validation.json` | `e5e63c9b8fc34f112c6f32ef994447efc6e0e3356ff921c41544a853466e4ebd` | Approval: `962f274bd4009a47ea0349691000e7c778fc1a8cf36b6e2c67434a1d1d4e46d1`; final freeze: `dc6fd4b4ba725a2517cd8e8bb5ed6b095584571e050dd349b5583fa225bfd356` |

The frozen question set SHA-256 is `ec7c151ba709eb6e55d37290dc8fb349a5def8b204f439b8e354f71d0b97d599`, its configuration SHA-256 is `bae9572fdc3007b857e7d8af3c2e01a0ef0f5b9b8767f187c8de5b31771007c6`, and the solutions manual SHA-256 is `d4a78d8cd1c2ae218741e372a36422882d5e804e3fa15215e494c67054aed62d`. The source-linkage commit is `0691263c28b31f5376be18e50c2bba316f0c1d04`.

We keep private backups of the original files and approved metadata beside the working references. We froze the reference gate on 2026-09-24. We commit this public audit and the notebook before loading generated responses.
