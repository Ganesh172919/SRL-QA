Run the RAISE fast deterministic QA check:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python -m srlqa.main ask --no-model --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```

Run the legacy baseline checkpoint check:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
python main.py --mode ask --engine baseline --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```

## 3. Streamlit Demo Apps

Launch the new SRL + RAG explainable QA app:

```powershell
streamlit run srl_rag_demo\app.py
```

Use this app for the main demo because it shows:

- PropBank corpus/index status
- Pasted/uploaded document retrieval
- SRL role evidence
- Final answer with confidence
- Explainable semantic graph and graph JSON export

Launch the newer RAISE-SRL-QA app:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
streamlit run raise_streamlit_app.py
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```

Launch the legacy PropQA-Net research app:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
python main.py --mode app --port 8502
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```

## 4. RAISE-SRL-QA Experiments

Show active configuration:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python -m srlqa.main show-config
```

Build or refresh the PropBank frame index from the legacy NLTK data:

```powershell
python -m srlqa.main build-frame-index
```

Preview the configured QA-SRL dataset through the `datasets` library:

```powershell
python -m srlqa.main preview-data --max-examples 5
```

Run the challenge-suite demo without loading the model-backed QA teacher:

```powershell
python -m srlqa.main demo --max-examples 8 --no-model
```

Run the model-backed RAISE demo if local weights/network are available:

```powershell
python -m srlqa.main demo --max-examples 8
```

Compare all available model families on one example:

```powershell
python run_all_models.py --model all --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?" --expected-answer "to the office"
```

Return to the workspace root:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```

## 5. Legacy PropQA-Net Experiments

Ask with the saved baseline checkpoint:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
python main.py --mode ask --engine baseline --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"
```

Ask with the legacy hybrid system:

```powershell
python main.py --mode ask --engine hybrid --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"
```

Run the legacy inference demo:

```powershell
python main.py --mode infer
```

Run the legacy benchmark on a smaller demo sample:

```powershell
python main.py --mode benchmark --max-examples 25
```

Run legacy evaluation against the saved checkpoint:

```powershell
python main.py --mode eval
```

Return to the workspace root:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```

## 6. Full Regeneration Commands

Use these only when you have time to regenerate artifacts. They may take longer and may update generated outputs.

Current checkout note: `srl_qa_project\main.py` has `full` and `report` branches that call `generate_all_pdfs`, but the `pdf_generator.py` source file is not present in this checkout. The old PDF outputs already exist under `srl_qa_project\outputs\`. For a live demo, prefer the smoke checks, Streamlit apps, RAISE demos, legacy `ask`, legacy `infer`, legacy `benchmark`, and legacy `eval` commands above.

Legacy full train/eval/infer/benchmark/report pipeline:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
python main.py --mode full --max-examples 160
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```

Run this only after restoring or re-adding the missing PDF generator module.

Legacy training only:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
python main.py --mode train
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```

Legacy report generation only:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
python main.py --mode report
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```

Run this only after restoring or re-adding the missing PDF generator module.

## 7. Notebook And Artifact References

These are reference artifacts for presentation and comparison, not required for the fast live demo:

- `PropBank_SRL_QA_2B_Gemma_QLoRA.ipynb`
- `PropBank_SRL_QA_LoRA_QLoRA.ipynb`
- `propbank_srlqa_2b_artifacts\research_summary_2b.txt`
- `propbank_srlqa_artifacts\research_summary.txt`
- `propbank_srlqa_2b_artifacts\plots\`
- `propbank_srlqa_artifacts\plots\`
- `srlqa\RAISE_SRLQA_PLOTS_FULL_PRESENTATION.pptx`
- `srlqa\plots\RAISE_SRLQA_ACCURATE_EXPANDED_PRESENTATION.pptx`

## 8. Recommended Live Demo Order

1. Run `python srl_rag_demo\smoke_test.py`.
2. Launch `streamlit run srl_rag_demo\app.py`.
3. Ask: `Where was the package delivered?`
4. Use pasted text: `The courier delivered the package to the office at noon.`
5. Open the retrieved evidence tab.
6. Open the explainable graph tab and download the graph JSON.
7. Optionally run the RAISE comparison:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py --model all --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?" --expected-answer "to the office"
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
```
