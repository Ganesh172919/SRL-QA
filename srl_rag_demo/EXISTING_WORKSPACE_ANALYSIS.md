# Existing Workspace Analysis

## `srl_qa_project`

- Legacy PropQA-Net project with a classical BiLSTM-style SRL + QA checkpoint.
- Loads real PropBank through NLTK from `srl_qa_project/nltk_data`.
- Local corpus check found 112,917 PropBank instances visible through NLTK and
  9,353 Treebank-backed instances that can be aligned to local Treebank parses.
- Existing checkpoint: `srl_qa_project/checkpoints/best_model.pt`.
- Existing cached splits: `data/val.json` and `data/test.json`.
- `data/train.json` is missing in the current workspace, so this demo does not
  depend on the legacy cached train split.

## `srlqa`

- Cleaner RAISE-SRL-QA scaffold with deterministic SRL-QA, PropBank frame
  retrieval, verifier scoring, and a Streamlit demo.
- The fast RAISE pipeline answered the sample question "Where was the package
  delivered?" with "to the office" using PropBank frame evidence.
- Existing frame store: `srlqa/retrieval/frame_store.json`.
- The frame store contains 4,659 PropBank frame records and is reused by this
  standalone demo for frame hints and role compatibility checks.

## Experiment Artifacts

- `PropBank_SRL_QA_2B_Gemma_QLoRA.ipynb`
- `PropBank_SRL_QA_LoRA_QLoRA.ipynb`
- `propbank_srlqa_2b_artifacts/`
- `propbank_srlqa_artifacts/`

These are LoRA/Gemma experiment notebooks and generated artifacts. They are kept
as references only and are not required for the local SRL + RAG demo path.
