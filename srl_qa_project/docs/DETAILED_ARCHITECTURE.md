# Detailed Architecture: PropQA-Net and Hybrid SRL-QA System

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Data Pipeline Architecture](#3-data-pipeline-architecture)
4. [PropQA-Net Model Architecture](#4-propqa-net-model-architecture)
5. [Training Architecture](#5-training-architecture)
6. [Evaluation Architecture](#6-evaluation-architecture)
7. [Inference Architecture](#7-inference-architecture)
8. [Hybrid QA Architecture](#8-hybrid-qa-architecture)
9. [Benchmark Architecture](#9-benchmark-architecture)
10. [Streamlit App Architecture](#10-streamlit-app-architecture)
11. [Configuration Architecture](#11-configuration-architecture)
12. [File and Module Organization](#12-file-and-module-organization)
13. [Data Flow Diagrams](#13-data-flow-diagrams)
14. [Component Interactions](#14-component-interactions)
15. [Deployment Architecture](#15-deployment-architecture)
16. [Extensibility Points](#16-extensibility-points)

---

## 1. System Overview

The PropQA-Net system is a modular, multi-component architecture for semantic role-based question answering. It consists of three major layers:

1. **Core Pipeline**: Data processing, model training, and evaluation
2. **Hybrid Inference Layer**: Role-aware reranking, transformer QA assist, and semantic matching
3. **Research Interface**: Streamlit dashboard, benchmark runner, and PDF deliverables

Each layer is independently testable and can be used without the others, enabling flexible deployment.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RESEARCH INTERFACE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Streamlit   │  │  Benchmark   │  │   PDF Deliverables       │  │
│  │     App      │  │   Runner     │  │   (survey, analysis,     │  │
│  │  (app.py)    │  │(benchmark.py)│  │   innovation, paper)     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┬───────────┘  │
│         │                 │                          │              │
├─────────┼─────────────────┼──────────────────────────┼──────────────┤
│         │                 │     HYBRID LAYER         │              │
│         ▼                 ▼                          │              │
│  ┌──────────────────────────────────────┐            │              │
│  │        Hybrid QA System              │            │              │
│  │  ┌──────────┐ ┌──────────┐ ┌───────┐│            │              │
│  │  │ Baseline │ │Heuristic │ │Transf.││            │              │
│  │  │ (PropQA) │ │ Extract  │ │  QA   ││            │              │
│  │  └────┬─────┘ └────┬─────┘ └───┬───┘│            │              │
│  │       └────────────┴──────────┴────┘│            │              │
│  │              │                      │            │              │
│  │       ┌──────▼──────┐               │            │              │
│  │       │  Semantic   │               │            │              │
│  │       │  Reranker   │               │            │              │
│  │       └─────────────┘               │            │              │
│  │  (hybrid_qa.py)                     │            │              │
│  └──────────────────┬──────────────────┘            │              │
│                     │                               │              │
├─────────────────────┼───────────────────────────────┼──────────────┤
│                     │       CORE PIPELINE           │              │
│                     ▼                               │              │
│  ┌──────────────────────────────────────────────────┼──────────┐  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────▼───────┐  │  │
│  │  │   Data   │ │  Model   │ │ Training │ │  Evaluation   │  │  │
│  │  │ Pipeline │ │PropQA-Net│ │  Loop    │ │  & Error Ana. │  │  │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬───────┘  │  │
│  │       │             │           │               │          │  │
│  │  ┌────▼─────────────▼───────────▼───────────────▼──────┐   │  │
│  │  │              Inference Engine                        │   │  │
│  │  │            (qa_inference.py)                         │   │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌───────────────────────────────────────────────────────────────┐│
│  │                  Configuration Layer                          ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐││
│  │  │  Paths   │ │   Data   │ │  Model   │ │    Training      │││
│  │  │  Config  │ │  Config  │ │  Config  │ │     Config       │││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘││
│  │                    (config.py)                               ││
│  └──────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipeline Architecture

### 3.1 Module: `data_loader.py`

The data pipeline transforms raw PropBank annotations into QA training examples.

### 3.2 Pipeline Stages

```
┌─────────────────────┐
│  NLTK PropBank      │
│  + Treebank subset  │
│  (nltk_data/)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Initialize NLTK    │
│  Register local     │
│  nltk_data path     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Load PropBank      │
│  Instances          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Filter: Treebank   │
│  Alignment Check    │
│  (usable only)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Build Visible      │
│  Token View         │
│  (exclude -NONE-)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Resolve Pointers   │
│  -> Token Indices   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Assign BIO Tags    │
│  for Each Role      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Generate Questions │
│  from Role Templates│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Build Example Dict │
│  (context, question,│
│   answer, SRL tags) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Deterministic      │
│  Split (70/15/15)   │
│  Seed = 42          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Cache to JSON      │
│  (data/*.json)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Build Vocabularies │
│  (token, POS, label)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Create DataLoaders │
│  (collate + pad)    │
└─────────────────────┘
```

### 3.3 Key Classes and Functions

| Component | Type | Purpose |
|-----------|------|---------|
| `Vocabulary` | Class | String-to-index mapping for tokens, POS tags, labels |
| `SRLQADataset` | Class (Dataset) | Encodes examples into tensors |
| `initialize_nltk()` | Function | Registers local NLTK data path |
| `build_visible_token_view()` | Function | Creates token/POS view from Treebank tree |
| `visible_indices_for_pointer()` | Function | Resolves PropBank pointer to token indices |
| `assign_bio_labels()` | Function | Writes BIO tags for argument spans |
| `build_question()` | Function | Generates natural-language question from role |
| `build_examples_from_propbank()` | Function | Main pipeline: PropBank -> examples |
| `split_examples()` | Function | Deterministic train/val/test split |
| `build_dataloaders()` | Function | Creates DataLoaders with collation |
| `collate_batch()` | Function | Pads variable-length examples into batches |

### 3.4 Example JSON Schema

Each cached example contains:

```json
{
  "example_id": "qa_000001",
  "instance_id": "wsj_0001.mrg:0:0:run-01",
  "fileid": "wsj_0001.mrg",
  "sentnum": 0,
  "context": "The chef cooked a delicious meal in the kitchen yesterday.",
  "context_tokens": ["The", "chef", "cooked", "a", "delicious", "meal", "in", "the", "kitchen", "yesterday", "."],
  "question": "Who cooked?",
  "question_tokens": ["Who", "cooked", "?"],
  "answer_text": "The chef",
  "answer_tokens": ["The", "chef"],
  "answer_start": 0,
  "answer_end": 1,
  "answer_length": 2,
  "predicate_lemma": "cook",
  "predicate_text": "cooked",
  "predicate_indices": [2],
  "predicate_flags": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  "roleset_id": "run-01",
  "roleset_name": "run",
  "target_role": "ARG0",
  "target_role_description": "agent",
  "question_type": "WHO",
  "pos_tags": ["DT", "NN", "VBD", "DT", "JJ", "NN", "IN", "DT", "NN", "NN", "."],
  "ne_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "DATE", "O"],
  "dependency_labels": ["det", "nsubj", "root", "det", "amod", "obj", "case", "det", "obl:loc", "obl:tmod", "punct"],
  "srl_tags": ["B-ARG0", "I-ARG0", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
  "argument_spans": [...]
}
```

---

## 4. PropQA-Net Model Architecture

### 4.1 Module: `model.py`

### 4.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PropQA-Net                                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    CONTEXT ENCODER PATH                       │  │
│  │                                                              │  │
│  │  context_ids ──> [Word Emb 100d] ──┐                        │  │
│  │  pos_ids ─────> [POS Emb 32d]  ────+──> [Concat 140d]      │  │
│  │  pred_flags ──> [Pred Emb 8d]  ────┘         │              │  │
│  │                                              ▼              │  │
│  │                                    [BiLSTM 128h x2]         │  │
│  │                                    (context_hidden=256)     │  │
│  │                                              │              │  │
│  │                                    [Dropout 0.30]           │  │
│  │                                              │              │  │
│  │                          ┌───────────────────┼───────────┐  │  │
│  │                          ▼                   ▼           │  │  │
│  │                  [SRL Classifier]    [Interaction Layer]  │  │  │
│  │                  (Linear 256->N_lbl)                      │  │  │
│  │                          │                                │  │  │
│  │                          ▼                                │  │  │
│  │                  srl_logits                               │  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   QUESTION ENCODER PATH                       │  │
│  │                                                              │  │
│  │  question_ids ──> [Word Emb 100d] (shared)                   │  │
│  │                          │                                   │  │
│  │                          ▼                                   │  │
│  │                  [BiLSTM 128h x2]                            │  │
│  │                  (question_hidden=256)                       │  │
│  │                          │                                   │  │
│  │                          ▼                                   │  │
│  │                  [Masked Mean Pooling]                       │  │
│  │                          │                                   │  │
│  │                          ▼                                   │  │
│  │                  [Question Projection]                       │  │
│  │                  (Linear 256->256)                           │  │
│  │                          │                                   │  │
│  │                          ▼                                   │  │
│  │                  question_vector (batch, 256)                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   SPAN SCORING PATH                           │  │
│  │                                                              │  │
│  │  context_outputs (batch, seq_len, 256)                       │  │
│  │  question_vector (batch, 256)                                │  │
│  │                                                              │  │
│  │  interaction = [                                               │  │
│  │    context_outputs,              # (batch, seq, 256)         │  │
│  │    question_expanded,            # (batch, seq, 256)         │  │
│  │    context_outputs * question,   # (batch, seq, 256)         │  │
│  │    |context_outputs - question|  # (batch, seq, 256)         │  │
│  │  ] -> (batch, seq, 1024)                                     │  │
│  │                                                              │  │
│  │  start_logits = Linear(1024->1)(interaction) -> (batch, seq) │  │
│  │  end_logits   = Linear(1024->1)(interaction) -> (batch, seq) │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      DECODING                                 │  │
│  │                                                              │  │
│  │  1. Decode BIO spans from srl_logits.argmax()                │  │
│  │  2. For each candidate span:                                  │  │
│  │     - Compute span vector (mean of context outputs)           │  │
│  │     - Cosine similarity with question_vector                  │  │
│  │     - Boundary confidence from start/end logits               │  │
│  │     - score = 0.60 * cosine + 0.40 * boundary                │  │
│  │  3. Select highest-scoring span                               │  │
│  │  4. Fallback to best boundary if no clean BIO spans           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `context_ids` | (batch, max_ctx_len) | Token IDs for context |
| `pos_ids` | (batch, max_ctx_len) | POS tag IDs |
| `predicate_flags` | (batch, max_ctx_len) | Binary predicate indicators |
| `context_mask` | (batch, max_ctx_len) | Boolean padding mask |
| `question_ids` | (batch, max_q_len) | Token IDs for question |
| `question_mask` | (batch, max_q_len) | Boolean padding mask |
| `context_outputs` | (batch, max_ctx_len, 256) | Contextualized token states |
| `question_vector` | (batch, 256) | Pooled question representation |
| `srl_logits` | (batch, max_ctx_len, num_labels) | SRL classification logits |
| `start_logits` | (batch, max_ctx_len) | Start position scores |
| `end_logits` | (batch, max_ctx_len) | End position scores |

### 4.4 Key Functions

| Function | Purpose |
|----------|---------|
| `PropQANet.__init__()` | Initialize all model components |
| `PropQANet._encode_lstm()` | Encode with BiLSTM and packing |
| `PropQANet.encode_context()` | Full context encoding pipeline |
| `PropQANet.encode_question()` | Full question encoding pipeline |
| `PropQANet.forward()` | Complete forward pass with loss computation |
| `PropQANet.predict()` | Decode predictions for a batch |
| `PropQANet.model_summary()` | Return model statistics |
| `decode_bio_spans()` | Decode BIO tags into argument spans |
| `masked_mean_pooling()` | Compute mask-aware mean pooling |
| `strip_bio_prefix()` | Remove B-/I- prefix from labels |
| `majority_role()` | Find most frequent non-O role in a window |

---

## 5. Training Architecture

### 5.1 Module: `trainer.py`

### 5.2 Training Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                           │
│                                                             │
│  1. Set random seed (42)                                    │
│  2. Initialize PropQA-Net from vocabularies                 │
│  3. Initialize Adam optimizer (lr=1e-3, wd=1e-5)           │
│  4. For epoch in 1..max_epochs (6):                         │
│     │                                                       │
│     ├── TRAINING PHASE                                      │
│     │   For each batch in train_loader:                     │
│     │     1. Move batch to device (CPU/GPU)                 │
│     │     2. Forward pass -> loss                           │
│     │     3. Zero gradients                                 │
│     │     4. Backward pass                                  │
│     │     5. Clip gradients (norm=5.0)                      │
│     │     6. Optimizer step                                 │
│     │     7. Record loss                                    │
│     │                                                       │
│     ├── VALIDATION PHASE                                    │
│     │   For each batch in val_loader:                       │
│     │     1. Move batch to device                           │
│     │     2. Forward pass (no grad)                         │
│     │     3. Decode predictions                             │
│     │     4. Compute EM and F1                              │
│     │     5. Record loss, EM, F1                            │
│     │                                                       │
│     ├── CHECKPOINT PHASE                                    │
│     │   If val_f1 > best_val_f1:                            │
│     │     1. Save checkpoint                                │
│     │     2. Reset patience counter                          │
│     │   Else:                                               │
│     │     1. Increment patience counter                      │
│     │     2. If patience >= 5: early stop                   │
│     │                                                       │
│     └── LOG PHASE                                           │
│         Print epoch metrics                                 │
│                                                             │
│  5. Load best checkpoint                                    │
│  6. Return model and training summary                       │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Checkpoint Structure

```json
{
  "model_state": "<torch state_dict>",
  "config": "<project configuration dict>",
  "history": [{"epoch": 1.0, "train_loss": ..., "validation_loss": ..., "validation_em": ..., "validation_f1": ...}],
  "best_epoch": 3,
  "best_validation_f1": 0.85,
  "vocabularies": {
    "token_vocab": {"token_to_id": {...}, "id_to_token": [...]},
    "pos_vocab": {"token_to_id": {...}, "id_to_token": [...]},
    "label_vocab": {"token_to_id": {...}, "id_to_token": [...]}
  },
  "model_summary": {"name": "PropQA-Net", "trainable_parameters": ..., ...}
}
```

### 5.4 Key Functions

| Function | Purpose |
|----------|---------|
| `set_random_seed()` | Seed Python, NumPy, and Torch RNGs |
| `move_batch_to_device()` | Move tensors to target device |
| `token_level_f1()` | Compute token-overlap F1 for answers |
| `evaluate_validation_split()` | Evaluate val loss, EM, and F1 |
| `serialize_vocabularies()` | Convert vocab objects to dicts |
| `train_model()` | Main training loop with checkpointing |

---

## 6. Evaluation Architecture

### 6.1 Module: `evaluator.py`

### 6.2 Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                       │
│                                                             │
│  1. Load best checkpoint                                    │
│  2. Rebuild PropQA-Net from checkpoint vocabularies         │
│  3. Load model state dictionary                             │
│  4. Set model to eval mode                                  │
│                                                             │
│  5. Generate prediction records                             │
│     For each batch in test_loader:                          │
│       1. Move batch to device                               │
│       2. Decode predictions                                 │
│       3. For each prediction:                               │
│          - Extract predicted text and tokens                │
│          - Compute exact match                              │
│          - Compute token F1                                 │
│          - Record role prediction                           │
│          - Record confidence                                │
│                                                             │
│  6. Compute SRL metrics                                     │
│     - Per-role precision/recall/F1                          │
│     - Macro and micro averages                              │
│     - BIO accuracy                                          │
│     - Confusion matrix                                      │
│                                                             │
│  7. Compute QA metrics                                      │
│     - Overall EM and F1                                     │
│     - Per-question-type breakdown                           │
│     - Answer length deviation                               │
│                                                             │
│  8. Perform error analysis                                  │
│     - Classify errors (correct, predicate miss, wrong role, │
│       span boundary, other)                                 │
│     - Error rates by sentence length                        │
│     - Error rates by role                                   │
│     - Top 20 worst errors                                   │
│                                                             │
│  9. Generate plots                                          │
│     - Loss curve                                            │
│     - Per-role F1 bar chart                                 │
│     - Confusion matrix heatmap                              │
│     - QA accuracy by question type                          │
│     - Answer length distribution                            │
│     - Error taxonomy pie chart                              │
│                                                             │
│  10. Save metrics to JSON                                   │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Key Functions

| Function | Purpose |
|----------|---------|
| `load_trained_model()` | Load checkpoint and rebuild model |
| `prediction_records()` | Generate per-example prediction records |
| `role_metrics_from_records()` | Compute SRL metrics from records |
| `qa_metrics_from_records()` | Compute QA metrics from records |
| `classify_error()` | Assign error category to a record |
| `error_analysis()` | Produce error-focused summaries |
| `plot_loss_curve()` | Plot training/validation loss |
| `plot_role_f1()` | Plot per-role F1 scores |
| `plot_confusion()` | Plot confusion matrix heatmap |
| `plot_qtype_metrics()` | Plot EM and F1 by question type |
| `plot_answer_length_distribution()` | Plot answer length histograms |
| `plot_error_taxonomy()` | Plot error classification pie chart |
| `evaluate_model()` | Main evaluation orchestrator |

---

## 7. Inference Architecture

### 7.1 Module: `qa_inference.py`

### 7.2 Inference Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                        │
│                                                             │
│  Input: context (string), question (string)                 │
│                                                             │
│  1. Tokenize context (regex-based)                          │
│  2. Tokenize question                                       │
│  3. Assign heuristic POS tags                               │
│  4. Infer predicate index                                   │
│     - Match lemmatized question tokens in context           │
│     - Fall back to verb-like tokens (-ed, -ing)             │
│     - Fall back to token index 1                            │
│  5. Build predicate flags tensor                            │
│  6. Encode context tokens with vocabulary                   │
│  7. Encode question tokens with vocabulary                  │
│  8. Encode POS tags with POS vocabulary                     │
│  9. Build batch tensors (with padding)                      │
│  10. Run model.predict()                                    │
│  11. Extract answer tokens from context                     │
│  12. Return answer text, confidence, predicted role         │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Key Classes and Functions

| Component | Type | Purpose |
|-----------|------|---------|
| `InferenceOutput` | Dataclass | Container for inference results |
| `InferenceEngine` | Class | Runtime wrapper around trained model |
| `simple_word_tokenize()` | Function | Regex-based tokenization |
| `simple_lemmatize()` | Function | Approximate lemmatization |
| `heuristic_pos_tags()` | Function | Lightweight POS tagging |
| `infer_predicate_index()` | Function | Find predicate token in context |
| `ask_question()` | Function | Run one custom question |
| `run_interactive_session()` | Function | Terminal QA loop |
| `run_demo()` | Function | 10-example inference demo |
| `demo_examples()` | Function | Return demo example list |

---

## 8. Hybrid QA Architecture

### 8.1 Module: `hybrid_qa.py`

### 8.2 Hybrid System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Hybrid QA System                              │
│                                                                     │
│  Input: context (string), question (string)                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: Question Intent Analysis                             │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  - Detect WH-word -> question_type                     │  │  │
│  │  │  - Map to expected_role                                │  │  │
│  │  │  - Extract predicate_hint                              │  │  │
│  │  │  - Extract target_terms                                │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  STEP 2: Baseline Prediction                                  │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  PropQA-Net inference -> (answer, role, confidence)    │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: Candidate Generation                                 │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │  │
│  │  │   Baseline   │ │  Heuristic   │ │   Transformer QA     │  │  │
│  │  │   Candidate  │ │  Candidates  │ │   Candidates (opt)   │  │  │
│  │  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘  │  │
│  │         └────────────────┴────────────────────┘              │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │                    [Deduplication]                            │  │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                   │
│                                 ▼                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  STEP 4: Candidate Scoring                                    │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  For each candidate:                                    │  │  │
│  │  │    - semantic_alignment (sentence embeddings or lexical)│  │  │
│  │  │    - role_match_score (expected vs candidate role)      │  │  │
│  │  │    - lexical_overlap (Jaccard with target terms)        │  │  │
│  │  │    - shape_bonus (surface form appropriateness)         │  │  │
│  │  │    - baseline_bonus (agreement with baseline)           │  │  │
│  │  │    - final_score = weighted combination                 │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                   │
│                                 ▼                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  STEP 5: Best Candidate Selection                             │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  - Prefer exact role match                             │  │  │
│  │  │  - Fall back to highest score                          │  │  │
│  │  │  - Fall back to baseline if no candidates              │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                   │
│                                 ▼                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  STEP 6: Reasoning Trace Generation                           │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  - Deterministic template-based explanation            │  │  │
│  │  │  - Optional LLM rewrite (Flan-T5, env-gated)           │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                 │                                   │
│                                 ▼                                   │
│  Output: HybridPrediction (answer, role, confidence, reasoning,    │
│           evidence_spans, baseline_comparison, diagnostics)        │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 Heuristic Extractor Details

| Extractor | Target Role | Pattern |
|-----------|-------------|---------|
| `_agent_span()` | ARG0 | Noun phrase before predicate |
| `_theme_span()` | ARG1 | Noun phrase after predicate |
| `_recipient_spans()` | ARG2 | After "to"/"for" prepositions |
| `_temporal_spans()` | ARGM-TMP | Time markers and temporal prepositions |
| `_location_spans()` | ARGM-LOC | Location prepositions + noun phrases |
| `_manner_spans()` | ARGM-MNR | "-ly" adverbs, instrumental phrases |
| `_cause_spans()` | ARGM-CAU | "because", "due to" constructions |

### 8.4 External Model Bundle

| Model | Purpose | Loading Condition |
|-------|---------|-------------------|
| DistilBERT (SQuAD) | Transformer QA span proposals | `use_transformer_qa=True` |
| SentenceTransformers (all-MiniLM-L6-v2) | Semantic similarity scoring | `use_sentence_embeddings=True` |
| Flan-T5-small | Reasoning trace rewriting | `SRL_QA_ENABLE_REASONER=1` |

### 8.5 Key Classes and Functions

| Component | Type | Purpose |
|-----------|------|---------|
| `QuestionIntent` | Dataclass | Parsed question intent |
| `CandidateSpan` | Dataclass | Answer candidate with features |
| `HybridPrediction` | Dataclass | Structured hybrid result |
| `ExternalModelBundle` | Class | Lazy loader for optional models |
| `HybridQASystem` | Class | Main hybrid QA orchestrator |
| `load_challenge_suite()` | Function | Load curated challenge examples |
| `sample_questions()` | Function | Return sample questions for app |
| `evaluate_prediction()` | Function | Attach accuracy metrics to prediction |

---

## 9. Benchmark Architecture

### 9.1 Module: `benchmark.py`

### 9.2 Benchmark Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    BENCHMARK PIPELINE                        │
│                                                             │
│  1. Load test examples and challenge suite                  │
│  2. Sample question-type-aware test subset                  │
│                                                             │
│  3. Initialize all tracks:                                  │
│     - Classical Baseline (PropQA-Net only)                  │
│     - Heuristic Reranker (baseline + heuristics)            │
│     - Transformer QA Assist (baseline + transformer)        │
│     - Full Hybrid (all components)                          │
│                                                             │
│  4. For each track:                                         │
│     a. Evaluate on challenge suite                          │
│     b. Evaluate on test subset                              │
│     c. Combine records                                      │
│     d. Aggregate metrics (EM, F1, role accuracy, latency)   │
│     e. Compute per-question-type and per-role breakdowns    │
│                                                             │
│  5. Generate benchmark artifacts:                           │
│     - Ablation summary plot                                 │
│     - Latency-accuracy tradeoff plot                        │
│     - Question type heatmap                                 │
│     - Role heatmap                                          │
│     - Confidence histogram                                  │
│     - Dataset balance plot                                  │
│     - Challenge table                                       │
│     - Error gallery                                         │
│     - Research architecture diagram                         │
│                                                             │
│  6. Save benchmark results to JSON                          │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 Key Functions

| Function | Purpose |
|----------|---------|
| `sample_test_examples()` | Question-type-aware sampling |
| `run_benchmark()` | Main benchmark orchestrator |
| `_evaluate_track()` | Evaluate one track over examples |
| `_aggregate_records()` | Aggregate records into summary metrics |
| `generate_benchmark_artifacts()` | Generate all benchmark plots |
| `attach_benchmark_to_metrics()` | Merge benchmark into metrics payload |
| `load_latest_benchmark()` | Load saved benchmark results |

---

## 10. Streamlit App Architecture

### 10.1 Module: `app.py`

### 10.2 App Sections

| Section | Purpose |
|---------|---------|
| Ask the Model | Interactive QA with sample questions |
| Architecture | Visual system diagrams |
| Dataset & PropBank Explorer | Data statistics and distributions |
| Experiments | Benchmark results and track comparison |
| Tradeoffs | Latency vs. accuracy analysis |
| Documentation | Project walkthrough and research anchors |
| Downloads | Export metrics, benchmarks, and PDFs |

### 10.3 Key Functions

| Function | Purpose |
|----------|---------|
| `get_hybrid_system()` | Cached hybrid system (Streamlit cache) |
| `highlight_answer()` | Highlight answer span in context |
| `render_ask_section()` | QA interface with sample questions |
| `render_architecture_section()` | Architecture diagram display |
| `render_dataset_section()` | Dataset statistics explorer |
| `render_experiments_section()` | Benchmark results dashboard |
| `render_tradeoffs_section()` | Trade-off analysis |
| `render_documentation_section()` | Long-form documentation |
| `render_downloads_section()` | Download buttons for artifacts |

---

## 11. Configuration Architecture

### 11.1 Module: `config.py`

### 11.2 Configuration Hierarchy

```
ProjectConfig
├── PathConfig
│   ├── project_root
│   ├── data_dir
│   ├── checkpoints_dir
│   ├── results_dir
│   ├── plots_dir
│   ├── outputs_dir
│   ├── nltk_data_dir
│   └── ... (derived paths)
├── DataConfig
│   ├── random_seed (42)
│   ├── train_ratio (0.70)
│   ├── validation_ratio (0.15)
│   ├── test_ratio (0.15)
│   ├── lowercase_tokens (True)
│   ├── min_token_frequency (1)
│   ├── max_sentence_length (128)
│   ├── max_question_length (32)
│   └── ...
├── ModelConfig
│   ├── word_embedding_dim (100)
│   ├── pos_embedding_dim (32)
│   ├── predicate_embedding_dim (8)
│   ├── hidden_size (128)
│   ├── question_hidden_size (128)
│   ├── dropout (0.30)
│   └── alpha (0.50)
├── TrainingConfig
│   ├── batch_size (64)
│   ├── learning_rate (1e-3)
│   ├── weight_decay (1e-5)
│   ├── max_epochs (6)
│   ├── patience (5)
│   ├── grad_clip_norm (5.0)
│   └── num_workers (0)
└── RuntimeConfig
    ├── device ("cpu" or "cuda")
    └── verbose (True)
```

---

## 12. File and Module Organization

```
srl_qa_project/
│
├── main.py                 # CLI orchestrator (train/eval/infer/ask/benchmark/report/app/full)
├── config.py               # Centralized configuration (PathConfig, DataConfig, ModelConfig, etc.)
├── data_loader.py          # Data pipeline (PropBank -> examples -> DataLoaders)
├── model.py                # PropQA-Net architecture (BiLSTM + SRL head + span heads)
├── trainer.py              # Training loop (Adam, early stopping, checkpointing)
├── evaluator.py            # Evaluation pipeline (metrics, plots, error analysis)
├── qa_inference.py         # Inference engine (raw-text QA, demo, interactive session)
├── hybrid_qa.py            # Hybrid QA system (role-aware reranking, multi-channel)
├── benchmark.py            # Benchmark runner (four-track evaluation, ablation plots)
├── app.py                  # Streamlit research dashboard
├── requirements.txt        # Python dependencies
│
├── nltk_data/              # Bundled NLTK corpora (PropBank + Treebank subset)
├── data/                   # Cached processed splits (train.json, val.json, test.json)
├── checkpoints/            # Model checkpoint (best_model.pt)
├── results/                # Evaluation outputs
│   ├── metrics.json        # SRL + QA metrics
│   ├── data_statistics.json # Dataset statistics
│   ├── inference_demo.json  # Demo results
│   ├── plots/              # Evaluation plots (PNG)
│   └── benchmarks/         # Benchmark results
│       └── benchmark_results.json
├── outputs/                # PDF deliverables
│   ├── survey.pdf
│   ├── analysis.pdf
│   ├── innovation.pdf
│   ├── research_paper.pdf
│   └── implementation_code.py
└── docs/                   # Documentation
    ├── OVERVIEW.md
    ├── DATA.md
    ├── MODEL.md
    ├── EVALUATION.md
    ├── PDF_DELIVERABLES.md
    ├── TROUBLESHOOTING.md
    ├── COMPLETE_PROJECT_GUIDE.md
    ├── DETAILED_ANALYSIS.md
    ├── DETAILED_INNOVATION.md
    ├── DETAILED_SURVEY.md
    └── DETAILED_ARCHITECTURE.md
```

---

## 13. Data Flow Diagrams

### 13.1 Training Data Flow

```
PropBank Instances ──> Treebank Alignment ──> Visible Token View
       │                                              │
       │                                              ▼
       │                                     BIO Tag Assignment
       │                                              │
       │                                              ▼
       │                                     Question Generation
       │                                              │
       │                                              ▼
       │                                     Example Dictionary
       │                                              │
       │                                              ▼
       │                                     Deterministic Split
       │                                    ┌─────┬─────┬─────┐
       │                                    │Train│ Val │Test │
       │                                    └──┬──┴──┬──┴──┬──┘
       │                                       │     │     │
       │                                       ▼     │     │
       │                                 Vocabularies │     │
       │                                       │     │     │
       │                                       ▼     │     │
       │                                 DataLoaders  │     │
       │                                       │     │     │
       │                                       ▼     │     │
       │                                 PropQA-Net   │     │
       │                                 Training     │     │
       │                                       │     │     │
       │                                       ▼     │     │
       │                                 Checkpoint   │     │
       │                                       │     │     │
       │                                       └─────┼─────┘
       │                                             │
       │                                             ▼
       │                                     Evaluation
       │                                             │
       │                                             ▼
       │                                     Metrics + Plots
```

### 13.2 Inference Data Flow

```
Context (string) ──> Tokenization ──> Encoding ──┐
                                                  │
Question (string) -> Tokenization -> Encoding ────┤
                                                  ▼
                                          PropQA-Net Forward
                                                  │
                                                  ▼
                                          Prediction Decoding
                                                  │
                                                  ▼
                                          Answer + Role + Confidence
```

### 13.3 Hybrid Inference Data Flow

```
Context + Question
       │
       ▼
  Question Intent Analysis
       │
       ▼
  ┌──────────────────────────────────┐
  │     Candidate Generation         │
  │  ┌────────┐ ┌─────────┐ ┌─────┐│
  │  │Baseline│ │Heuristic│ │Trans││
  │  └───┬────┘ └────┬────┘ └──┬──┘│
  │      └───────────┴─────────┘   │
  │              │                  │
  │              ▼                  │
  │       Deduplication             │
  └──────────────┬──────────────────┘
                 │
                 ▼
         Semantic Reranking
                 │
                 ▼
         Best Candidate Selection
                 │
                 ▼
         Reasoning Trace Generation
                 │
                 ▼
         HybridPrediction Output
```

---

## 14. Component Interactions

### 14.1 Module Dependencies

```
main.py
├── config.py
├── data_loader.py
├── model.py (indirect via trainer)
├── trainer.py
├── evaluator.py
├── qa_inference.py
├── hybrid_qa.py
└── benchmark.py

trainer.py
├── config.py
└── model.py

evaluator.py
├── config.py
├── model.py
└── trainer.py (token_level_f1, move_batch_to_device)

qa_inference.py
├── config.py
└── evaluator.py (load_trained_model, normalize_text)

hybrid_qa.py
├── config.py
├── evaluator.py (normalize_text)
├── qa_inference.py (InferenceEngine, tokenization, lemmatization)
└── trainer.py (token_level_f1)

benchmark.py
├── config.py
├── evaluator.py (normalize_text)
├── hybrid_qa.py (HybridQASystem, load_challenge_suite)
├── qa_inference.py (InferenceEngine, tokenization)
└── trainer.py (token_level_f1)

app.py
├── config.py
├── benchmark.py (load_latest_benchmark)
└── hybrid_qa.py (HybridQASystem, load_challenge_suite)
```

### 14.2 Shared Utilities

| Utility | Defined In | Used By |
|---------|-----------|---------|
| `normalize_text()` | evaluator.py | hybrid_qa.py, benchmark.py |
| `token_level_f1()` | trainer.py | hybrid_qa.py, benchmark.py |
| `simple_word_tokenize()` | qa_inference.py | hybrid_qa.py, benchmark.py |
| `simple_lemmatize()` | qa_inference.py | hybrid_qa.py |
| `move_batch_to_device()` | trainer.py | evaluator.py |

---

## 15. Deployment Architecture

### 15.1 Local Development

```
┌──────────────────────────────────────┐
│         Local Machine                │
│                                      │
│  ┌────────────────────────────────┐  │
│  │  srl_qa_project/               │  │
│  │  ├── main.py (CLI)             │  │
│  │  ├── app.py (Streamlit)        │  │
│  │  ├── nltk_data/                │  │
│  │  ├── data/                     │  │
│  │  ├── checkpoints/              │  │
│  │  └── results/                  │  │
│  └────────────────────────────────┘  │
│                                      │
│  Python 3.10+                        │
│  PyTorch (CPU or CUDA)               │
│  Optional: transformers,             │
│            sentence-transformers     │
└──────────────────────────────────────┘
```

### 15.2 Execution Modes

| Mode | Components Used | Output |
|------|----------------|--------|
| `train` | data_loader, model, trainer | Checkpoint |
| `eval` | data_loader, model, evaluator | Metrics + plots |
| `infer` | data_loader, model, qa_inference | Demo results |
| `ask` | qa_inference or hybrid_qa | Answer to custom question |
| `benchmark` | hybrid_qa, benchmark | Benchmark results |
| `report` | evaluator, benchmark | PDF deliverables |
| `app` | app.py (Streamlit) | Web dashboard |
| `full` | All components | All outputs |

---

## 16. Extensibility Points

### 16.1 Data Pipeline Extensions

- **New data sources**: Add loaders for other semantic corpora (FrameNet, VerbNet)
- **Cross-lingual support**: Extend to other languages with PropBank annotations
- **Document-level data**: Support multi-sentence contexts

### 16.2 Model Extensions

- **Transformer encoder**: Replace BiLSTM with BERT/RoBERTa
- **Subword tokenization**: Add WordPiece/BPE for OOV handling
- **Additional heads**: Add predicate identification, dependency parsing

### 16.3 Hybrid System Extensions

- **New heuristic extractors**: Add role-specific extractors
- **New candidate sources**: Add retrieval-based or knowledge-based candidates
- **Custom reranking weights**: Optimize weights for specific domains

### 16.4 Evaluation Extensions

- **New metrics**: Add BLEU, ROUGE, or task-specific metrics
- **New benchmarks**: Add domain-specific challenge suites
- **Human evaluation**: Integrate human judgment collection

### 16.5 Interface Extensions

- **API server**: Deploy as REST API (FastAPI, Flask)
- **Batch processing**: Support bulk question answering
- **Visualization**: Add interactive architecture diagrams

---

*This detailed architecture document provides a comprehensive view of the PropQA-Net system, from high-level design to component-level details, data flows, and extensibility points. It serves as a reference for understanding, maintaining, and extending the system.*
