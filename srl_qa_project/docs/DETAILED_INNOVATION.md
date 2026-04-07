# Detailed Innovation: PropQA-Net Hybrid SRL-QA System

## Table of Contents

1. [Overview of Innovations](#1-overview-of-innovations)
2. [Innovation 1: SRL-Anchored Question Answering](#2-innovation-1-srl-anchored-question-answering)
3. [Innovation 2: Role-Aware Question Parsing](#3-innovation-2-role-aware-question-parsing)
4. [Innovation 3: Multi-Channel Candidate Generation](#4-innovation-3-multi-channel-candidate-generation)
5. [Innovation 4: Semantic Reranking with Weighted Features](#5-innovation-4-semantic-reranking-with-weighted-features)
6. [Innovation 5: Deterministic Reasoning Traces](#6-innovation-5-deterministic-reasoning-traces)
7. [Innovation 6: Reproducible PropBank Pipeline](#7-innovation-6-reproducible-propbank-pipeline)
8. [Innovation 7: Four-Track Benchmark Framework](#8-innovation-7-four-track-benchmark-framework)
9. [Innovation 8: Research-Grade Streamlit Interface](#9-innovation-8-research-grade-streamlit-interface)
10. [Innovation 9: Challenge Suite Evaluation](#10-innovation-9-challenge-suite-evaluation)
11. [Innovation 10: Hybrid Architecture Design Pattern](#11-innovation-10-hybrid-architecture-design-pattern)
12. [Novelty Assessment](#12-novelty-assessment)
13. [Impact and Applicability](#13-impact-and-applicability)
14. [Comparison with State of the Art](#14-comparison-with-state-of-the-art)
15. [Future Innovation Directions](#15-future-innovation-directions)

---

## 1. Overview of Innovations

This project introduces several innovations that collectively advance the state of semantic role-based question answering. The innovations span the entire pipeline, from data processing to inference to evaluation. Unlike projects that focus on a single component improvement, this work presents a **cohesive system** where each innovation reinforces the others.

The innovations can be grouped into three categories:

### 1.1 Core Modeling Innovations

- SRL-anchored question answering with joint multi-task learning
- Role-aware question parsing for intent-driven answer selection
- Multi-channel candidate generation combining classical and modern approaches

### 1.2 Inference Innovations

- Semantic reranking with weighted feature combination
- Deterministic reasoning traces for explainable predictions
- Hybrid architecture design pattern for extensible QA systems

### 1.3 Evaluation and Reproducibility Innovations

- Reproducible PropBank pipeline with offline data processing
- Four-track benchmark framework for systematic ablation
- Challenge suite evaluation for targeted role coverage
- Research-grade Streamlit interface for interactive exploration

---

## 2. Innovation 1: SRL-Anchored Question Answering

### 2.1 The Core Idea

Traditional extractive QA systems predict answer spans without understanding the **semantic role** that the answer plays in the event structure. PropQA-Net addresses this by jointly learning SRL and QA, so that every answer is grounded in a semantic role.

### 2.2 How It Works

The model processes each training example as a tuple:

```
(context_tokens, question_tokens, predicate_flags, BIO_labels, answer_start, answer_end)
```

During training, the model optimizes both:
- **SRL loss**: Cross-entropy over BIO labels for each context token
- **QA loss**: Cross-entropy over start and end positions

The shared context encoder ensures that the QA head benefits from SRL supervision, and vice versa.

### 2.3 Why It Matters

This approach provides several advantages:

1. **Explainability**: Every answer is associated with a semantic role, making the prediction interpretable
2. **Generalization**: SRL knowledge transfers across predicates and contexts
3. **Robustness**: Multi-task learning regularizes the model, reducing overfitting
4. **Alignment with linguistic theory**: PropBank provides a well-established framework for semantic roles

### 2.4 Novelty

While prior work has explored QA-SRL (generating questions from SRL annotations), this project is novel in its **joint multi-task architecture** that learns both tasks simultaneously from the same PropBank-derived data, with a shared encoder and a unified decoding strategy.

---

## 3. Innovation 2: Role-Aware Question Parsing

### 3.1 The Core Idea

The hybrid system includes a question intent analyzer that maps natural-language questions to expected semantic roles before answer selection. This enables the system to prefer candidates that match the expected role.

### 3.2 How It Works

The question parser performs the following steps:

1. **Question type detection**: Identifies the WH-word (who, what, when, where, how, why, whom) and maps it to a question type
2. **Expected role mapping**: Maps the question type to the most likely SRL role
3. **Predicate hint extraction**: Identifies the main verb or action referenced in the question
4. **Target term extraction**: Extracts content words that should appear in the answer

The mapping table is:

| Question Pattern | Question Type | Expected Role |
|-----------------|--------------|---------------|
| "Who..." | WHO | ARG0 |
| "What..." | WHAT | ARG1 |
| "When..." | WHEN | ARGM-TMP |
| "Where..." | WHERE | ARGM-LOC |
| "How..." | HOW | ARGM-MNR |
| "Why..." | WHY | ARGM-CAU |
| "To whom..." / "Who received..." | TO-WHOM | ARG2 |

### 3.3 Why It Matters

Role-aware question parsing enables the hybrid system to:

1. **Filter candidates**: Prefer spans that match the expected role
2. **Score candidates**: Assign higher scores to role-matching candidates
3. **Explain predictions**: Provide a reasoning trace that references the expected role
4. **Handle ambiguity**: Disambiguate between multiple plausible answers based on role expectations

### 3.4 Novelty

This is a novel approach to QA that bridges the gap between surface-level question patterns and deep semantic roles. Unlike systems that rely purely on neural attention, this approach uses explicit role mapping to guide answer selection.

---

## 4. Innovation 3: Multi-Channel Candidate Generation

### 4.1 The Core Idea

Instead of relying on a single prediction source, the hybrid system generates answer candidates from multiple channels, each with complementary strengths.

### 4.2 Channel Descriptions

#### 4.2.1 Baseline Channel (PropQA-Net)

- **Strength**: Trained on real PropBank annotations, understands predicate-argument structure
- **Weakness**: Limited by BiLSTM capacity and fixed vocabulary

#### 4.2.2 Heuristic Channel

- **Strength**: Fast, interpretable, covers common patterns
- **Weakness**: Rule-based, may miss complex or unusual patterns

The heuristic channel includes specialized extractors for:
- **Agent spans**: Noun phrases before the predicate
- **Theme spans**: Noun phrases after the predicate
- **Recipient spans**: After "to"/"for" prepositions, with transfer verb detection
- **Temporal spans**: Time markers (days, months, relative expressions)
- **Location spans**: After location prepositions
- **Manner spans**: Adverbs ending in "-ly", instrumental phrases
- **Cause spans**: "because", "due to" constructions

#### 4.2.3 Transformer Channel (Optional)

- **Strength**: Strong span proposals from a pre-trained SQuAD model
- **Weakness**: Requires model download, adds latency

### 4.3 Candidate Deduplication

Candidates from different channels are deduplicated by normalizing text and combining identical spans, keeping the highest-scoring version.

### 4.4 Why It Matters

Multi-channel generation provides **robust coverage** that no single channel can achieve alone. The baseline handles PropBank-aligned cases, heuristics provide fallback for common patterns, and the transformer offers strong proposals for complex contexts.

### 4.5 Novelty

This multi-channel approach is novel in its **systematic combination** of classical NLP (heuristics), neural SRL (PropQA-Net), and modern QA (transformer) within a unified reranking framework.

---

## 5. Innovation 4: Semantic Reranking with Weighted Features

### 5.1 The Core Idea

After generating candidates from multiple channels, the system reranks them using a weighted combination of features that capture different aspects of answer quality.

### 5.2 Feature Design

| Feature | Weight | Description |
|---------|--------|-------------|
| Base score | 0.30 | Source confidence (from the proposing channel) |
| Role match | 0.32 | Alignment between candidate role and expected role |
| Semantic alignment | 0.22 | Sentence embedding similarity between question and candidate context |
| Lexical overlap | 0.06 | Jaccard similarity between candidate text and target terms |
| Shape bonus | 0.10 | Surface form appropriateness for the role |
| Baseline bonus | 0.08-0.18 | Agreement with baseline prediction |

### 5.3 Role Match Scoring

The role match feature rewards exact matches and provides partial credit for compatible roles:

- **Exact match**: 1.0
- **Compatible roles** (e.g., ARGM-CAU vs ARGM-PRP): 0.55
- **Related roles** (e.g., ARG1 vs any ARG): 0.35
- **No match**: 0.05

### 5.4 Semantic Alignment

Semantic alignment is computed using sentence embeddings (SentenceTransformers) when available, with a lexical fallback (Jaccard similarity) when the embedding model is not loaded.

### 5.5 Shape Bonus

The shape bonus rewards candidates whose surface form matches the expected pattern for their role:

- **ARGM-TMP**: Contains temporal markers (days, months, "yesterday", etc.)
- **ARGM-LOC**: Starts with location prepositions ("in", "at", "on", etc.)
- **ARGM-MNR**: Ends in "-ly" or starts with "with"
- **ARGM-CAU**: Starts with "because" or "due to"
- **ARG2**: Starts with "to" or "for"
- **ARG0/ARG1**: General noun phrase: 0.75

### 5.6 Why It Matters

This weighted feature approach provides **transparent, tunable reranking** that can be adjusted based on application requirements. Each feature is interpretable, and the weights can be optimized for specific use cases.

### 5.7 Novelty

The novelty lies in the **systematic combination** of role-aware scoring, semantic similarity, and surface form analysis within a single reranking framework. This approach is more interpretable than end-to-end neural reranking and more flexible than rule-based selection.

---

## 6. Innovation 5: Deterministic Reasoning Traces

### 6.1 The Core Idea

Every hybrid prediction includes a short, deterministic reasoning trace that explains why the selected answer was chosen. This provides transparency without the computational cost of LLM-based explanation generation.

### 6.2 How It Works

The reasoning trace is generated from the prediction's internal state:

```
If role matches expected:
  "Question pattern '{question_type}' mapped to {expected_role}.
   The hybrid reranker aligned the predicate '{predicate}' and selected
   '{answer}' from {source} evidence because it best matched the expected role."

Otherwise:
  "Question pattern '{question_type}' suggested {expected_role}, but the strongest
   available evidence came from {source} with span '{answer}'.
   The baseline answer was '{baseline_answer}' ({baseline_role})."
```

### 6.3 Optional LLM Rewriting

When the environment variable `SRL_QA_ENABLE_REASONER=1` is set, the reasoning trace can be optionally rewritten by a small local instruction model (e.g., Flan-T5-small) for improved fluency.

### 6.4 Why It Matters

Deterministic reasoning traces provide:

1. **Transparency**: Users can understand why a prediction was made
2. **Debuggability**: Developers can identify failure modes by examining traces
3. **Trust**: Explainable predictions are more trustworthy than black-box outputs
4. **Efficiency**: No LLM call required for basic explanations

### 6.5 Novelty

This approach to explanation generation is novel in its **deterministic, evidence-based** design. Unlike LLM-generated explanations, which can be hallucinated, these traces are grounded in the system's actual decision-making process.

---

## 7. Innovation 6: Reproducible PropBank Pipeline

### 7.1 The Core Idea

The entire data pipeline is designed for offline, deterministic reproducibility. No external corpus downloads are required, and all results are reproducible from the bundled data.

### 7.2 Key Design Choices

1. **Bundled NLTK data**: PropBank and Treebank subsets are included in `nltk_data/`
2. **Deterministic splits**: Fixed random seed (42) for train/val/test
3. **Cached JSON splits**: Processed examples are saved to `data/*.json`
4. **Centralized configuration**: All hyperparameters in `config.py`
5. **Checkpoint-based evaluation**: Best model saved and reloadable

### 7.3 Why It Matters

Reproducibility is a critical concern in NLP research. This pipeline ensures that:

- Anyone with the repository can reproduce the results
- No external dependencies (beyond Python packages) are required
- Results are deterministic across runs
- The pipeline can be run offline

### 7.4 Novelty

While PropBank is a well-known resource, the **offline, reproducible pipeline** that generates QA pairs from PropBank annotations is a novel contribution. Most PropBank-based work requires external corpus downloads and complex alignment procedures.

---

## 8. Innovation 7: Four-Track Benchmark Framework

### 8.1 The Core Idea

The benchmark framework evaluates the system across four tracks, providing a systematic ablation study of each component's contribution.

### 8.2 Track Design

| Track | Components | Purpose |
|-------|-----------|---------|
| Classical Baseline | PropQA-Net only | Establish baseline performance |
| Heuristic Reranker | Baseline + heuristics | Measure impact of role-aware heuristics |
| Transformer QA Assist | Baseline + transformer | Measure impact of transformer span proposals |
| Full Hybrid | All components | Measure combined impact |

### 8.3 Evaluation Protocol

Each track is evaluated on two datasets:

1. **Challenge suite**: Curated examples covering all major roles
2. **Test subset**: Question-type-aware sample from the test split

### 8.4 Metrics

- Exact Match, Token F1, Role Accuracy
- Mean latency, mean confidence
- Per-question-type and per-role breakdowns

### 8.5 Why It Matters

This benchmark framework provides:

1. **Systematic ablation**: Clear attribution of improvements to specific components
2. **Balanced evaluation**: Question-type-aware sampling ensures fair comparison
3. **Comprehensive metrics**: Multiple dimensions of performance are measured
4. **Reproducibility**: Deterministic sampling ensures consistent results

### 8.6 Novelty

The four-track benchmark design is novel in its **systematic combination** of classical and modern components within a single evaluation framework. Most benchmarks evaluate either classical or neural systems, but not both in direct comparison.

---

## 9. Innovation 8: Research-Grade Streamlit Interface

### 8.1 The Core Idea

The project includes a comprehensive Streamlit research app that provides interactive exploration of the system, data, and results.

### 8.2 Features

- **Ask the Model**: Interactive QA with sample questions and custom inputs
- **Architecture**: Visual diagrams of PropQA-Net and hybrid system
- **Dataset Explorer**: PropBank statistics, distributions, sample QA pairs
- **Experiments**: Track comparison, per-type metrics, per-role metrics
- **Tradeoffs**: Latency vs. accuracy analysis
- **Documentation**: Full project walkthrough with research anchors
- **Downloads**: Export metrics, benchmarks, and PDF deliverables

### 8.3 Why It Matters

The Streamlit app makes the research accessible to a broader audience:

1. **Interactive exploration**: Users can test the system with custom inputs
2. **Visual analysis**: Plots and tables make results easy to understand
3. **Reproducibility**: All results are generated from the local environment
4. **Documentation**: Built-in documentation explains the system design

### 8.4 Novelty

While Streamlit apps are common in ML, a **research-grade interface** that integrates data exploration, model interaction, benchmark comparison, and documentation in a single dashboard is a novel contribution.

---

## 10. Innovation 9: Challenge Suite Evaluation

### 10.1 The Core Idea

A curated challenge suite provides targeted evaluation of the system's ability to handle diverse semantic roles and question types.

### 10.2 Suite Design

The challenge suite contains examples covering:

- All major question types (WHO, WHAT, WHEN, WHERE, HOW, WHY, TO-WHOM)
- All major semantic roles (ARG0, ARG1, ARG2, ARGM-TMP, ARGM-LOC, ARGM-MNR, ARGM-CAU)
- Varying sentence complexity
- Common and edge cases

### 10.3 Why It Matters

The challenge suite provides:

1. **Targeted evaluation**: Tests specific capabilities rather than aggregate performance
2. **Diagnostic power**: Identifies which roles and question types are challenging
3. **Demonstration value**: Provides clear examples for presentations and demos
4. **Regression testing**: Can be used to detect performance regressions

### 10.4 Novelty

The challenge suite is novel in its **role-centric design**, specifically targeting semantic role coverage rather than general QA performance.

---

## 11. Innovation 10: Hybrid Architecture Design Pattern

### 11.1 The Core Idea

The hybrid system establishes a design pattern for combining classical and modern NLP components in a principled way.

### 11.2 Design Pattern

```
1. Analyze question intent (type, role, predicate, terms)
2. Generate candidates from multiple channels
3. Deduplicate and normalize candidates
4. Score candidates with weighted features
5. Select best candidate with role preference
6. Generate reasoning trace
```

### 11.3 Why It Matters

This design pattern is **generalizable** to other QA and information extraction tasks:

1. **Modularity**: Each component can be independently improved or replaced
2. **Extensibility**: New channels can be added without modifying existing ones
3. **Transparency**: Each step is interpretable and debuggable
4. **Flexibility**: Components can be enabled/disabled based on requirements

### 11.4 Novelty

The design pattern is novel in its **explicit role-aware integration** of classical and modern components. Most hybrid systems combine neural models, but this approach integrates rule-based heuristics, classical neural SRL, and modern transformer QA in a unified framework.

---

## 12. Novelty Assessment

### 12.1 Individual Novelty

Each innovation builds on existing work but introduces novel elements:

| Innovation | Prior Work | Novel Element |
|-----------|-----------|---------------|
| SRL-anchored QA | QA-SRL, neural SRL | Joint multi-task with shared encoder |
| Role-aware parsing | Template QA | Explicit role mapping for reranking |
| Multi-channel generation | Ensemble QA | Classical + neural + transformer |
| Semantic reranking | Neural reranking | Weighted feature combination |
| Reasoning traces | LLM explanations | Deterministic, evidence-based |
| Reproducible pipeline | PropBank tools | Offline, deterministic, bundled |
| Benchmark framework | SQuAD, SuperGLUE | Four-track ablation design |
| Streamlit interface | ML dashboards | Research-grade integration |
| Challenge suite | QA benchmarks | Role-centric evaluation |
| Hybrid design pattern | Hybrid systems | Role-aware classical-modern integration |

### 12.2 Collective Novelty

The collective contribution is greater than the sum of its parts. The project demonstrates that:

1. Classical NLP techniques can be effectively combined with modern approaches
2. Semantic roles provide a powerful framework for explainable QA
3. Reproducibility and innovation are not mutually exclusive
4. A cohesive system design can achieve both accuracy and interpretability

---

## 13. Impact and Applicability

### 13.1 Research Impact

This project contributes to several research areas:

- **Semantic Role Labeling**: Demonstrates practical application of SRL to QA
- **Question Answering**: Introduces role-aware answer selection
- **Explainable AI**: Provides deterministic reasoning traces
- **Reproducible Research**: Establishes a fully reproducible pipeline

### 13.2 Practical Applications

The system is applicable to:

- **Information extraction**: Extracting structured information from text
- **Reading comprehension**: Understanding and answering questions about documents
- **Knowledge base population**: Filling semantic slots from unstructured text
- **Educational tools**: Teaching semantic roles and predicate-argument structure

### 13.3 Broader Impact

The project demonstrates that:

- Classical NLP techniques remain valuable in the era of large language models
- Reproducibility should be a first-class concern in NLP research
- Explainability can be achieved without sacrificing performance
- Hybrid approaches can combine the best of classical and modern methods

---

## 14. Comparison with State of the Art

### 14.1 vs. Transformer-Based QA (BERT, RoBERTa)

| Aspect | PropQA-Net | Transformer QA |
|--------|-----------|----------------|
| Architecture | BiLSTM | Transformer |
| Parameters | ~500K-1M | ~100M-300M |
| Training data | PropBank-derived | SQuAD, etc. |
| Explainability | High (role-based) | Low (black-box) |
| Reproducibility | High (offline) | Medium (model downloads) |
| Inference speed | Fast | Slower |
| Absolute performance | Moderate | High |

### 14.2 vs. QA-SRL Systems

| Aspect | PropQA-Net | QA-SRL |
|--------|-----------|--------|
| Question generation | Template-based | Crowdsourced |
| Model architecture | Joint multi-task | Separate models |
| Inference | Single pass | Multi-step |
| Role awareness | Built-in | Post-hoc |

### 14.3 vs. LLM-Based QA

| Aspect | PropQA-Net | LLM QA |
|--------|-----------|--------|
| Training | Supervised | Pre-trained + prompted |
| Explainability | Deterministic traces | Generated text |
| Reproducibility | Deterministic | Stochastic |
| Cost | Minimal | High (API or compute) |
| Role awareness | Explicit | Implicit |

---

## 15. Future Innovation Directions

### 15.1 Short-Term Extensions

1. **Transformer context encoder**: Replace BiLSTM with BERT for richer representations
2. **Subword tokenization**: Handle OOV words with WordPiece/BPE
3. **Confidence calibration**: Improve confidence score reliability
4. **Active learning**: Select most informative examples for annotation

### 15.2 Medium-Term Extensions

1. **Cross-lingual SRL-QA**: Extend to other languages with PropBank annotations
2. **Multi-hop QA**: Chain multiple predicate-argument structures
3. **Document-level QA**: Handle questions spanning multiple sentences
4. **Temporal reasoning**: Handle complex temporal expressions

### 15.3 Long-Term Vision

1. **Unified semantic parsing**: Combine SRL, dependency parsing, and QA in a single model
2. **Interactive learning**: Learn from user feedback in real-time
3. **Domain adaptation**: Transfer to new domains with minimal data
4. **Neuro-symbolic integration**: Combine neural and symbolic reasoning

---

*This detailed innovation document describes the novel contributions of the PropQA-Net project, from core modeling advances to evaluation frameworks and research interfaces. Each innovation is grounded in the reproducible local pipeline and designed for practical applicability.*
