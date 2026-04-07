# Evaluation Protocol

## Claim Discipline

Do not claim 95% token-F1 until a frozen split and saved scorer produce it. Use
three separate labels:

- `current baseline`: existing `srl_qa_project/results/metrics.json`.
- `local challenge-suite score`: new `srlqa/data/challenge_suite_v2.json` or a
  future hidden local suite.
- `official public score`: only CoNLL/OntoNotes or another public benchmark with
  the official scorer and matching split.

## Frozen Inputs

The initial manifest is stored at `srlqa/results/eval_manifest.json`.

Frozen references:

- Existing test split: `srl_qa_project/data/test.json`
- Existing validation split: `srl_qa_project/data/val.json`
- Existing challenge suite: `srl_qa_project/data/challenge_suite.json`
- New seed challenge suite: `srlqa/data/challenge_suite_v2.json`

## Metrics

Report all of these for every ablation:

- exact match
- token F1
- role accuracy
- predicate accuracy when available
- expected calibration error
- latency
- per-question-type F1
- per-role F1
- bootstrap 95% confidence interval for token F1

## Ablation Table

Minimum leaderboard rows:

- baseline `PropQANet`
- current hybrid baseline
- MRC encoder only
- MRC plus PropBank retrieval
- MRC plus constrained decoder
- MRC plus verifier/self-correction
- MRC plus hard negatives
- distilled student
- calibrated ensemble

## No Leakage Rule

Do not tune thresholds, templates, frame prompts, hard negatives, or verifier
weights on the final frozen test records. Use validation or a separate dev
challenge suite for those decisions.
