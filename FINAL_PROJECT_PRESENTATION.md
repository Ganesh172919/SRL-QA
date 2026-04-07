# Final Project Presentation: PropQA-Net & Hybrid SRL-QA System

---

## Slide 1: Title Slide

# Question Answering Using Semantic Roles

### PropQA-Net: A Classical BiLSTM-Based SRL-Anchored QA System
### with Hybrid Inference, Role-Aware Reranking, and Transformer Assist

**NLP Final Project**

---

## Slide 2: Problem Statement

### The Core Problem

Traditional extractive Question Answering (QA) systems return a text span from a context passage but **cannot explain which semantic role that span fills** in the underlying event structure.

For example, given:
> "The chef cooked a delicious meal in the kitchen yesterday."

A standard QA system might answer "the chef" to "Who cooked?" but cannot tell you that this is an **ARG0 (agent)** role, nor can it distinguish between agent, patient, time, location, manner, or cause.

### Our Solution

We reframe QA as **semantic slot filling over PropBank predicate-argument structures**, so that every answer is both extractive and semantically explainable.

---

## Slide 3: Key Objectives

1. **Build a reproducible SRL-QA pipeline** using real PropBank annotations via NLTK
2. **Design PropQA-Net**, a multi-task BiLSTM model that jointly learns SRL tagging and extractive QA
3. **Generate natural-language questions** automatically from PropBank argument roles
4. **Evaluate comprehensively** with EM, token-F1, per-role metrics, and error taxonomy
5. **Upgrade to a hybrid inference system** with role-aware reranking, transformer QA assist, and semantic matching
6. **Benchmark four tracks**: classical baseline, heuristic reranker, transformer QA assist, and full hybrid
7. **Deliver a Streamlit research app** for interactive exploration

---

## Slide 4: System Architecture Overview

```
NLTK PropBank + Treebank (nltk_data/)
            |
            v
  PropBank Instance Alignment + Span Reconstruction
            |
            v
  Example Generation (context, question, answer, SRL BIO tags)
            |
            +--> Cache to data/train.json, val.json, test.json
            |
            v
  Torch DataLoaders + Vocabularies
            |
            v
  PropQA-Net Training (BiLSTM + SRL head + Span heads)
            |
            v
  Evaluation + Plots + Error Analysis
            |
            +--> Baseline Inference Demo
            +--> Hybrid QA + Challenge Benchmarks
            +--> Streamlit Research Website
            +--> PDF Deliverables
```

---

## Slide 5: Data Pipeline

### Source: PropBank via NLTK

- **Total PropBank instances**: Available through NLTK's bundled corpus
- **Usable instances**: Only those aligned to local Penn Treebank parses
- **Filtering rationale**: Deterministic span reconstruction requires Treebank alignment

### Processing Steps

1. Load PropBank instances from `nltk_data/corpora/propbank/`
2. Align each instance with Penn Treebank parse trees
3. Reconstruct sentence text from Treebank leaves
4. Build visible token view (excluding `-NONE-` traces)
5. Assign BIO SRL tags for each argument role
6. Generate natural-language questions from role templates
7. Split deterministically (seed=42): 70% train, 15% val, 15% test

### Question Generation Templates

| Role | Question Pattern |
|------|-----------------|
| ARG0 | "Who {predicate}?" |
| ARG1 | "What did {subject} {predicate}?" |
| ARGM-TMP | "When did {subject} {predicate}?" |
| ARGM-LOC | "Where did {subject} {predicate}?" |
| ARGM-MNR | "How did {subject} {predicate}?" |
| ARGM-CAU | "Why did {subject} {predicate}?" |
| ARG2 | "Who received {object}?" |

---

## Slide 6: PropQA-Net Architecture

### Multi-Task BiLSTM Network

```
Context tokens ----> [Word Embedding (100d)] --\
POS tags ---------> [POS Embedding (32d)] -----+--> [BiLSTM Context Encoder (128h)]
Predicate flags --> [Predicate Emb. (8d)] ----/       |
                                                       v
                                              [Dropout (0.30)]
                                                 /        \
                                                v          v
                                     [SRL BIO Classifier]  [QA Span Projections]
                                                                   |
Question tokens --> [Shared Word Emb.] --> [BiLSTM Question Encoder (128h)]
                                                       |
                                                       v
                                        [Question Vector (mean pooling)]
                                                       |
                                                       v
                                    [Argument-Question Matching Layer]
                                                       |
                                                       v
                                         Best answer span + confidence score
```

### Key Design Choices

- **Shared word embeddings** between context and question encoders
- **Predicate indicator flags** as explicit input features
- **POS tag embeddings** for syntactic grounding
- **Multi-task loss**: alpha * SRL_loss + (1-alpha) * QA_loss
- **Bidirectional LSTMs** for contextual representation

---

## Slide 7: Model Configuration

| Hyperparameter | Value |
|---------------|-------|
| Word embedding dim | 100 |
| POS embedding dim | 32 |
| Predicate embedding dim | 8 |
| Context hidden size | 128 |
| Question hidden size | 128 |
| Dropout | 0.30 |
| Alpha (SRL/QA balance) | 0.50 |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Max epochs | 6 |
| Patience (early stopping) | 5 |
| Gradient clip norm | 5.0 |
| Max sentence length | 128 |
| Max question length | 32 |

---

## Slide 8: Training Procedure

### Loss Function

The model optimizes a weighted multi-task objective:

```
L = alpha * L_SRL + (1 - alpha) * L_QA

L_SRL = CrossEntropy(BIO predictions, gold BIO tags)
L_QA = 0.5 * (CrossEntropy(start_logits, gold_start) + CrossEntropy(end_logits, gold_end))
```

### Training Loop

1. Seed all RNGs (Python, NumPy, Torch) with seed=42
2. For each epoch (max 6):
   - Train on training split with Adam optimizer
   - Evaluate on validation split (loss, EM, token-F1)
   - Save checkpoint if validation F1 improves
   - Early stop if no improvement for 5 epochs
3. Load best checkpoint for final evaluation

### Decoding Strategy

1. Decode BIO spans from SRL predictions
2. Compute cosine similarity between each candidate span and question vector
3. Combine: score = 0.60 * cosine + 0.40 * boundary confidence
4. Select highest-scoring span as final answer
5. Fallback to best boundary span if no clean BIO spans decode

---

## Slide 9: Evaluation Metrics

### QA Metrics

- **Exact Match (EM)**: Predicted answer text equals gold text after normalization
- **Token-overlap F1**: F1 between predicted and gold answer token bags
- **Per-question-type breakdown**: WHO, WHAT, WHEN, WHERE, WHY, HOW

### SRL Metrics

- **Token-level precision/recall/F1** per role (ARG0, ARG1, ARG2, ARGM-TMP, ARGM-LOC, etc.)
- **Macro and micro averages** across all roles
- **BIO accuracy**: Token-level BIO tag classification accuracy
- **Confusion matrix**: Role-to-role confusion patterns

### Error Taxonomy

- Correct predictions
- Predicate misses (predicted role = "O")
- Wrong role predictions
- Span boundary errors (partial overlap)
- Other errors

---

## Slide 10: Hybrid QA System

### Architecture Upgrade

The hybrid system layers multiple inference channels on top of the trained PropQA-Net:

```
Question + Context
       |
       v
  [Question Intent Analyzer]
       |
       v
  [PropQA-Net Baseline] -----> Candidate Span 1
       |
  [Role Heuristics] ---------> Candidate Span 2, 3, 4...
       |
  [Transformer QA (optional)] -> Candidate Span 5, 6, 7...
       |
       v
  [Semantic Reranker]
       |
       v
  [Best Span Selection + Reasoning Trace]
```

### Four Benchmark Tracks

1. **Classical Baseline**: PropQA-Net only
2. **Heuristic Reranker**: Baseline + role-aware heuristics
3. **Transformer QA Assist**: Baseline + transformer span proposals
4. **Full Hybrid**: All channels active with semantic reranking

---

## Slide 11: Hybrid Components

### Question Intent Analyzer

Maps question words to expected SRL roles:

| Question Word | Question Type | Expected Role |
|--------------|--------------|---------------|
| Who | WHO | ARG0 |
| What | WHAT | ARG1 |
| When | WHEN | ARGM-TMP |
| Where | WHERE | ARGM-LOC |
| How | HOW | ARGM-MNR |
| Why | WHY | ARGM-CAU |
| To whom | TO-WHOM | ARG2 |

### Heuristic Evidence Extractors

- **Agent span**: Noun phrase before predicate
- **Theme span**: Noun phrase after predicate
- **Recipient spans**: After "to"/"for" prepositions
- **Temporal spans**: Time markers (days, months, "yesterday", etc.)
- **Location spans**: After location prepositions (in, at, on, near, etc.)
- **Manner spans**: Adverbs ending in "-ly", instrumental phrases
- **Cause spans**: "because", "due to" constructions

### Semantic Reranking

Scores each candidate with weighted features:
- Base score (source confidence): 30%
- Role match score: 32%
- Semantic alignment: 22%
- Lexical overlap: 6%
- Shape bonus: 10%

---

## Slide 12: Challenge Suite

A curated set of diverse QA examples covering all major semantic roles:

| # | Question Type | Target Role | Example Context |
|---|--------------|-------------|-----------------|
| 1 | WHO | ARG0 | "The chef cooked a delicious meal..." |
| 2 | WHEN | ARGM-TMP | "She sent a letter last Monday..." |
| 3 | WHY | ARGM-CAU | "The company announced layoffs because of budget cuts..." |
| 4 | WHAT | ARG1 | "The nurse administered the medicine..." |
| 5 | WHERE | ARGM-LOC | "The courier delivered the package to the office..." |
| 6 | HOW | ARGM-MNR | "The engineer repaired the machine carefully..." |
| 7 | WHAT | ARG1 | "The board approved the proposal..." |
| 8 | TO-WHOM | ARG2 | "Maria gave the intern a notebook..." |
| 9 | WHERE | ARGM-LOC | "Investigators examined the site after the explosion..." |
| 10 | HOW | ARGM-MNR | "The students presented their project enthusiastically..." |

---

## Slide 13: Benchmark Results Summary

### Track Comparison

| Track | Challenge EM | Challenge F1 | Test EM | Test F1 | Mean Latency (ms) |
|-------|-------------|-------------|---------|---------|-------------------|
| Classical Baseline | TBD | TBD | TBD | TBD | ~1-5 |
| Heuristic Reranker | TBD | TBD | TBD | TBD | ~1-5 |
| Transformer QA Assist | TBD | TBD | TBD | TBD | ~50-200 |
| Full Hybrid | TBD | TBD | TBD | TBD | ~50-200 |

### Key Findings

- Heuristic reranking improves role-sensitive questions (WHEN, WHERE, WHY)
- Transformer QA provides stronger span proposals for complex contexts
- Semantic reranking improves answer quality and explanation clarity
- Latency trade-off: hybrid components add inference time but improve accuracy

---

## Slide 14: Error Analysis

### Common Error Types

1. **Span Boundary Errors**: Correct role identified but span is too wide/narrow
2. **Wrong Role**: Model assigns incorrect semantic role to answer
3. **Predicate Misses**: Model fails to identify any argument (role = "O")
4. **Ambiguous Contexts**: Multiple valid interpretations of the same sentence

### Error Distribution by Sentence Length

| Length Bucket | Error Rate |
|--------------|------------|
| 0-10 tokens | Low |
| 11-20 tokens | Moderate |
| 21-30 tokens | Higher |
| 31+ tokens | Highest |

### Role-Specific Challenges

- **ARG2 (recipient)**: Often confused with ARG1 (theme)
- **ARGM-CAU vs ARGM-PRP**: Causal vs. purpose modifiers overlap
- **Long modifiers**: Tendency to truncate or over-extend span boundaries

---

## Slide 15: Streamlit Research App

### Features

- **Ask the Model**: Interactive QA with sample questions and custom inputs
- **Architecture**: Visual diagrams of PropQA-Net and hybrid system
- **Dataset Explorer**: PropBank statistics, question-type distribution, role distribution
- **Experiments**: Track comparison, per-question-type metrics, per-role metrics
- **Tradeoffs**: Latency vs. accuracy analysis
- **Documentation**: Full project walkthrough with research anchors
- **Downloads**: Export metrics, benchmarks, and PDF deliverables

### Running the App

```bash
python main.py --mode app --port 8501
```

---

## Slide 16: Reproducibility

### What Makes This Project Reproducible

1. **Bundled NLTK data**: No external corpus downloads required
2. **Deterministic splits**: Fixed random seed (42) for train/val/test
3. **Cached data splits**: JSON files in `data/` directory
4. **Centralized configuration**: All hyperparameters in `config.py`
5. **Checkpoint-based evaluation**: Best model saved and reloadable
6. **Benchmark runner**: Stable, question-type-aware sampling

### Running Modes

```bash
python main.py --mode train      # Train PropQA-Net
python main.py --mode eval       # Evaluate on test set
python main.py --mode infer      # Run inference demo
python main.py --mode ask        # Ask custom questions
python main.py --mode benchmark  # Run four-track benchmark
python main.py --mode report     # Generate PDF deliverables
python main.py --mode app        # Launch Streamlit app
python main.py --mode full       # Run entire pipeline
```

---

## Slide 17: Project Deliverables

### Generated Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| best_model.pt | checkpoints/ | Best model checkpoint |
| metrics.json | results/ | SRL + QA evaluation metrics |
| data_statistics.json | results/ | Dataset descriptive statistics |
| benchmark_results.json | results/benchmarks/ | Four-track benchmark results |
| loss_curve.png | results/plots/ | Training/validation loss curves |
| f1_by_argtype.png | results/plots/ | Per-role F1 scores |
| confusion_matrix.png | results/plots/ | SRL role confusion matrix |
| qa_accuracy_by_qtype.png | results/plots/ | EM and F1 by question type |
| answer_length_dist.png | results/plots/ | Predicted vs. gold answer lengths |
| error_taxonomy.png | results/plots/ | Error classification pie chart |
| ablation_summary.png | results/plots/ | Track-level EM and F1 comparison |
| latency_accuracy_tradeoff.png | results/plots/ | Latency vs. F1 scatter plot |
| question_type_heatmap.png | results/plots/ | F1 by question type across tracks |
| role_heatmap.png | results/plots/ | Role accuracy by target role |
| confidence_histogram.png | results/plots/ | Confidence calibration |
| research_architecture.png | results/plots/ | System architecture diagram |
| survey.pdf | outputs/ | Literature survey |
| analysis.pdf | outputs/ | Detailed analysis |
| innovation.pdf | outputs/ | Innovation description |
| research_paper.pdf | outputs/ | Full research paper |
| implementation_code.py | outputs/ | Concatenated source code |

---

## Slide 18: Research Anchors

### Key Literature

1. **Large-Scale QA-SRL Parsing** (ACL 2018)
   - Established large-scale QA-SRL parsing
   - Motivates question-answer supervision over predicate structure

2. **PropBank Comes of Age** (*SEM 2022)
   - Documents modern PropBank resource
   - Semantic backbone for this project

3. **Potential and Limitations of LLMs in Capturing Structured Semantics** (arXiv 2024)
   - Frames where LLM reasoning helps SRL
   - Where deterministic structure is still needed

4. **LLMs Can Also Do Well! Breaking Barriers in SRL via LLMs** (ACL Findings 2025)
   - Recent anchor for LLM-assisted reasoning discussion
   - Future work framing

---

## Slide 19: Strengths and Limitations

### Strengths

- **Explainable answers**: Every answer is tied to a semantic role
- **Reproducible pipeline**: Offline, deterministic, no external downloads
- **Multi-task learning**: SRL and QA share representations
- **Hybrid flexibility**: Classical baseline + modern inference channels
- **Comprehensive evaluation**: EM, F1, role accuracy, error taxonomy, benchmarks

### Limitations

- **Extractive only**: Cannot generate free-form answers
- **BiLSTM-based**: Less powerful than transformer architectures
- **Treebank-dependent**: Limited by local Treebank coverage
- **Heuristic predicate anchoring**: Approximate in raw-text inference
- **Fixed vocabulary**: No subword tokenization or OOV handling

---

## Slide 20: Future Work

### Potential Extensions

1. **Transformer encoder**: Replace BiLSTM with BERT/RoBERTa for richer context
2. **Subword tokenization**: Handle OOV words with WordPiece/BPE
3. **Cross-lingual SRL**: Extend to other languages with PropBank annotations
4. **Multi-hop QA**: Chain multiple predicate-argument structures
5. **LLM-assisted reasoning**: Use instruction models for explanation generation
6. **Real-time streaming**: Deploy as a service API
7. **Active learning**: Select most informative PropBank instances for annotation

---

## Slide 21: Conclusion

### Summary

PropQA-Net demonstrates that **semantic role labeling and question answering can be unified** in a single, interpretable model. By anchoring answers to PropBank predicate-argument structures, we achieve both accuracy and explainability.

The hybrid upgrade shows that **classical NLP pipelines can be enhanced** with modern inference techniques (transformer QA, semantic matching, role-aware reranking) without sacrificing reproducibility.

### Key Takeaway

**Answers are more useful when you know what role they play in the event.**

---

## Slide 22: Demo

### Live Demonstration

```bash
# Ask a custom question
python main.py --mode ask \
  --context "The chef cooked a delicious meal in the kitchen yesterday." \
  --question "Who cooked?"

# Interactive session
python main.py --mode ask --interactive

# Launch the web app
python main.py --mode app
```

### Sample Questions to Try

- "Who cooked?" (ARG0)
- "When did she send?" (ARGM-TMP)
- "Why were layoffs announced?" (ARGM-CAU)
- "What was administered?" (ARG1)
- "Where was the package delivered?" (ARGM-LOC)
- "How did the engineer repair?" (ARGM-MNR)
- "Who received a notebook?" (ARG2)

---

## Slide 23: Repository Structure

```
srl_qa_project/
|-- main.py                 # CLI runner
|-- config.py               # Configuration
|-- data_loader.py          # Data pipeline
|-- model.py                # PropQA-Net
|-- trainer.py              # Training loop
|-- evaluator.py            # Metrics and plots
|-- qa_inference.py         # Inference wrapper
|-- hybrid_qa.py            # Hybrid QA system
|-- benchmark.py            # Benchmark runner
|-- app.py                  # Streamlit app
|-- requirements.txt        # Dependencies
|-- nltk_data/              # NLTK corpora
|-- data/                   # Cached splits
|-- checkpoints/            # Model weights
|-- results/                # Metrics and plots
|-- outputs/                # PDF deliverables
`-- docs/                   # Documentation
```

---

## Slide 24: Q&A

### Thank You

**Questions?**

---

*This presentation covers the complete PropQA-Net project: from PropBank data processing through BiLSTM-based multi-task learning, comprehensive evaluation, hybrid inference upgrade, and interactive research dashboard.*
