# What Is Next

Prepared on: 2026-04-07

This file answers the question: after making the project functional and
presentation-ready, what should be done next?

## 1. Immediate Next Step For Presentation

Use the project as a demo-first system.

1. Open `FINAL_PROJECT_PRESENTATION_RAISE_UPDATED.pptx`.
2. Keep `PROJECT_INDEX.md` and `COMPLETE_FUNCTIONAL_PROJECT_GUIDE.md` open.
3. Run this command before the presentation to warm up confidence:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py --model raise_srlqa_fast --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?" --expected-answer "to the office"
```

Say:

> The system returns `to the office` as `ARGM-LOC`, meaning the answer is the
> location role for the delivery event.

## 2. What Not To Do Tomorrow

Do not rerun heavy training or full benchmark during a live presentation.

Do not say:

> We achieved official 95% F1.

Say:

> 95% is a roadmap target. The implemented baseline score is 0.7612 QA token F1,
> and the new RAISE fast path reaches 1.0 token F1 on the 15-example local seed
> challenge suite.

## 3. Next Engineering Step

The highest-value engineering step is to turn the current RAISE demo into a
proper evaluation pipeline.

Do this in order:

| Order | Task | Why it matters |
|---:|---|---|
| 1 | Freeze a larger local test split | Prevents accidental tuning to the demo examples |
| 2 | Expand challenge suite from 15 to 300+ examples | Tests WHO, WHAT, WHEN, WHERE, WHY, HOW, ARG2, and boundary traps |
| 3 | Add a fast health-check command | Confirms demo functionality without loading every heavy model |
| 4 | Train the DeBERTa MRC SRL-QA model | Main path to real accuracy improvement |
| 5 | Evaluate with one saved script | Makes accuracy reproducible |
| 6 | Add error analysis tables | Shows exactly which roles still fail |
| 7 | Only then update result claims | Keeps the project honest and defensible |

## 4. Next Research Step

The best research direction is:

> Train a supervised MRC SRL-QA model, then combine it with retrieval,
> constrained decoding, and verifier correction.

Why:

- The baseline has useful semantic structure but weaker span precision.
- RAISE improves targeted demo behavior.
- A trained MRC reader can improve general span extraction.
- Retrieval and constraints help keep the model explainable and precise.

## 5. Next Accuracy Goal

Use these milestones:

| Stage | Realistic target |
|---|---:|
| Current baseline | `0.7612` QA token F1 |
| RAISE seed suite | `1.0` token F1 on 15 local examples |
| Expanded local challenge suite | Aim for `0.85+` first |
| Trained DeBERTa MRC model | Aim for `0.84-0.88` token F1 |
| MRC + retrieval + constrained decoding | Aim for `0.89-0.93` local token F1 |
| MRC + verifier + hard negatives | Only then attempt `0.94-0.95` local target |

Important:

> Do not claim a higher number until the script produces that number on a frozen
> split.

## 6. Best Project Story

Tell the project story like this:

1. The previous system built a working PropBank-based SRL-QA baseline.
2. The baseline produced useful metrics but had span-boundary and rare-role
   weaknesses.
3. The hybrid system improved demo answers with rules and reranking.
4. The new RAISE package adds retrieval, verification, recursive correction, and
   all-model comparison.
5. The project is now functional for demo and ready for the next real research
   step: larger evaluation and supervised MRC training.

## 7. One-Line Answer

If someone asks, "What is next?", answer:

> Next, we freeze a larger evaluation suite, train the DeBERTa MRC SRL-QA model,
> and report only reproducible accuracy from that frozen benchmark.
