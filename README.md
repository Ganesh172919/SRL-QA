# NLP Project

This workspace now contains two connected SRL-QA implementations:

- `srl_qa_project/`: the original PropQA-Net baseline, hybrid inference system,
  evaluation outputs, reports, and Streamlit dashboard.
- `srlqa/`: the new RAISE-SRL-QA package with PropBank retrieval, verifier-style
  correction, all-model comparison, and a separate Streamlit app.

Start here:

- `PROJECT_INDEX.md`
- `COMPLETE_FUNCTIONAL_PROJECT_GUIDE.md`
- `WHAT_NEXT.md`

Fastest demo:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py --model raise_srlqa_fast --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?" --expected-answer "to the office"
```

Compare all models:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py --model all --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?" --expected-answer "to the office"
```

Run the RAISE Streamlit app:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
streamlit run raise_streamlit_app.py
```

Run the original dashboard:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
streamlit run app.py
```

Accuracy wording:

> Baseline local QA token F1 is 0.7612. RAISE fast reaches 1.0 token F1 on the
> 15-example local seed challenge suite. That is a local demo-suite result, not
> an official public SOTA claim.
