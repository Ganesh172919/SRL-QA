# Professor Presentation Speech

## Purpose

This file gives a ready-to-speak explanation of the project for a professor, evaluator, or viva panel.

The speech explains:

- the problem,
- the survey motivation,
- the project architecture,
- the implementation work,
- the demo,
- the evaluation metrics,
- the contribution and limitations.

## 1. Short Opening Speech

Good morning respected professor.

My project is about building an explainable question answering system using Semantic Role Labeling, PropBank, and Retrieval-Augmented Question Answering.

The main idea is that a QA system should not only return an answer, but should also explain why that answer was selected. To do this, I used Semantic Role Labeling, or SRL, which identifies roles like who did the action, what was affected, where it happened, when it happened, and how it happened.

For example, in the sentence:

```text
The courier delivered the package to the office at noon.
```

the predicate is:

```text
delivered
```

and the semantic roles are:

```text
ARG0: The courier
ARG1: the package
ARGM-LOC: to the office
ARGM-TMP: at noon
```

So if the question is:

```text
Where was the package delivered?
```

the system can map the question to the role `ARGM-LOC` and answer:

```text
to the office
```

This makes the answer more explainable than a normal black-box QA model.

## 2. Full Professor Speech

Good morning respected professor.

Today I will explain my project on Semantic Role Labeling based Question Answering with RAG and explainable reasoning graphs.

The motivation for this project comes from a common limitation in question answering systems. Many QA systems can produce an answer, but they do not clearly explain the reasoning path behind the answer. In this project, I tried to make the QA process more structured and explainable by using Semantic Role Labeling.

Semantic Role Labeling identifies the meaning of different parts of a sentence around a predicate. For example, it identifies who performed an action, what object was affected, where the action happened, when it happened, and how it happened. These roles are represented using labels such as `ARG0`, `ARG1`, `ARGM-LOC`, `ARGM-TMP`, and `ARGM-MNR`.

The main dataset resource used in this project is PropBank. PropBank provides predicate-argument annotations, and I loaded the PropBank data locally through NLTK. I also used Treebank information to reconstruct readable sentence contexts. This allowed me to create QA examples and SRL-structured documents from real linguistic annotations.

The project has three main implementation parts.

First, there is the legacy baseline project in `srl_qa_project`. This part loads the PropBank and Treebank data, creates question-answer examples, and evaluates the original SRL-QA baseline model. It gives us the full baseline metrics. On the full PropBank-derived baseline test set, the legacy system achieves `51.84%` exact match, `76.12%` QA token F1, `71.33%` SRL micro F1, and `81.63%` BIO accuracy.

Second, there is the newer framework in `srlqa`. This framework adds more role-aware improvements. It includes PropBank frame retrieval, constrained answer selection, evidence verification, self-correction, and model comparison. In the controlled 15-example curated seed suite, the RAISE fast and model-assisted pipelines achieved `100.00%` exact match, `100.00%` token F1, and `100.00%` role accuracy. I treat this as a curated seed-suite result, not as a full-corpus claim.

Third, I created a new standalone Streamlit demo in `srl_rag_demo`. This is the final demo part of the project. It loads PropBank through NLTK, builds SRL-structured documents, supports pasted or uploaded user documents, performs retrieval using sentence-transformer embeddings if available, and falls back to TF-IDF retrieval if needed. Then it performs role-aware question answering and builds an explainable semantic graph using NetworkX and Plotly.

The final demo is designed to show the complete idea clearly:

```text
Retrieve SRL evidence -> select answer using semantic role -> show reasoning graph
```

For example, if I ask:

```text
Where was the package delivered?
```

the system retrieves the relevant sentence:

```text
The courier delivered the package to the office at noon.
```

It identifies the predicate:

```text
delivered
```

It maps the question type `Where` to the expected semantic role:

```text
ARGM-LOC
```

and it selects the answer:

```text
to the office
```

The graph explanation then connects the question, retrieved document, predicate, role, candidate span, and final answer. This is the explainability contribution of the project.

## 3. Architecture Explanation To Speak

The architecture of the project can be explained in four layers.

The first layer is the data layer. Here I use local NLTK data, mainly PropBank and Treebank. PropBank gives the predicate and argument annotations, while Treebank helps reconstruct readable sentences.

The second layer is the SRL processing layer. In this layer, each PropBank instance is converted into a structured record containing the sentence, predicate, roles, role spans, and answer candidates.

The third layer is the QA and retrieval layer. In the legacy system, the model directly predicts answer spans from SRL-based examples. In the newer systems, the project adds role-aware retrieval, frame information, answer verification, and hybrid reranking.

The fourth layer is the explanation layer. In the final Streamlit demo, the system builds a semantic graph. The graph shows the path:

```text
Question -> Retrieved Document -> Predicate -> Role -> Candidate Span -> Final Answer
```

This makes the reasoning process visible to the user.

## 4. Demo Explanation To Speak

For the live demo, I will run:

```powershell
streamlit run srl_rag_demo\app.py
```

In the Streamlit app, I can choose the PropBank sample size, optionally paste or upload documents, enter a question, choose the retrieval mode, and run QA.

The app has tabs for:

- final answer,
- corpus and index status,
- retrieved SRL evidence,
- explainable graph reasoning.

The demo does not require any external API key. It is local and CPU-friendly. If the embedding model is not available, it uses TF-IDF as a fallback retriever.

The smoke test verified that local NLTK can load `112,917` PropBank instances. In the demo loader, `9,353` Treebank-backed usable instances were found, and the smoke test successfully answered:

```text
to the office
```

for the question:

```text
Where was the package delivered?
```

The graph output had `12` nodes and `15` edges in the smoke test.

## 5. Evaluation Metrics Speech

For evaluation, I used several metrics.

Exact Match checks whether the predicted answer exactly matches the gold answer.

Token F1 gives partial credit when the predicted answer overlaps with the correct answer. This is useful because sometimes the model predicts an answer span that is partially correct but too long or too short.

Role Accuracy checks whether the predicted semantic role is correct. This is especially important in this project because the goal is not only to find an answer but also to ground it in the correct SRL role.

SRL Micro F1 measures the overall semantic role labeling quality across all roles.

BIO Accuracy measures token-level SRL sequence labeling accuracy using labels like `B-ARG0`, `I-ARG0`, and `O`.

Latency is used to measure how fast the system can answer, which matters for a live demo.

The main full-test baseline values are:

| Metric | Value |
|---|---:|
| QA exact match | 51.84% |
| QA token F1 | 76.12% |
| SRL micro F1 | 71.33% |
| SRL BIO accuracy | 81.63% |

For the controlled 15-example RAISE seed suite:

| System | Exact Match | Token F1 | Role Accuracy |
|---|---:|---:|---:|
| Legacy baseline | 20.00% | 55.22% | 33.33% |
| Legacy hybrid | 66.67% | 82.30% | 93.33% |
| RAISE-SRL-QA Fast | 100.00% | 100.00% | 100.00% |
| RAISE-SRL-QA Model | 100.00% | 100.00% | 100.00% |

The important point is that these results have different scopes. The legacy metrics are full-test baseline metrics. The RAISE `100.00%` result is a controlled curated seed-suite result. I do not mix these claims.

## 6. Innovation Speech

The main innovation of the project is that it connects SRL with retrieval and explainable QA.

Instead of retrieving only raw text, the system retrieves SRL-structured documents. These documents include the context, predicate, semantic roles, argument spans, and PropBank frame hints.

Then the QA system uses the question type to prefer the correct semantic role. For example, a `Where` question prefers `ARGM-LOC`, and a `When` question prefers `ARGM-TMP`.

Finally, the graph explanation visualizes the reasoning path. This makes the project more interpretable than a normal QA system that only returns a final text answer.

## 7. What We Have Done In This Project

In this project, I completed the following work:

| Area | Work Completed |
|---|---|
| Data loading | Loaded PropBank and Treebank through local NLTK data. |
| SRL processing | Converted predicate-argument annotations into structured QA evidence. |
| Baseline model | Preserved and documented the legacy PropQA-Net baseline. |
| Improved QA | Added and analyzed role-aware hybrid and RAISE-style systems. |
| RAG demo | Built a new standalone `srl_rag_demo` Streamlit application. |
| Retrieval | Implemented hybrid retrieval with embedding support and TF-IDF fallback. |
| Explainability | Built semantic graph explanations using NetworkX and Plotly. |
| Evaluation | Collected exact match, token F1, role accuracy, SRL F1, BIO accuracy, confidence, and latency metrics. |
| Documentation | Created multiple markdown files explaining architecture, metrics, key terms, flows, experiments, and presentation content. |

## 8. Contribution Speech

The contribution of my project is not only in training or evaluating one model. The contribution is a complete explainable QA pipeline.

The project connects:

```text
PropBank SRL data -> QA generation -> role-aware retrieval -> answer selection -> graph explanation
```

This gives a clearer reasoning path for QA. The system can show not only the answer, but also the predicate, role, retrieved evidence, and graph connection behind the answer.

## 9. Limitations Speech

There are also some limitations.

First, the RAISE `100.00%` result is from a small curated 15-example seed suite, so it should not be presented as a full-corpus result.

Second, the LoRA and QLoRA experiments are FAST_DEV runs, so they show implementation feasibility but not final paper-quality training.

Third, user-uploaded documents in the Streamlit demo use a simpler SRL-like parsing method. A future version could add a stronger SRL parser for arbitrary uploaded documents.

Fourth, optional transformer QA can be slower, so the default deterministic SRL path is better for a live demo.

## 10. Future Work Speech

In future work, I would improve the project in four ways.

First, I would run a larger controlled benchmark for the RAISE and SRL + RAG systems.

Second, I would add a stronger SRL parser for user-uploaded documents so the system can handle arbitrary text more accurately.

Third, I would run full-scale model training experiments with multiple seeds and compare models such as BERT, RoBERTa, DeBERTa, and Gemma under the same evaluation setup.

Fourth, I would improve the graph visualization by adding more detailed edge labels, confidence scores, and frame definitions.

## 11. Closing Speech

To conclude, this project builds an explainable QA system using Semantic Role Labeling. It starts from PropBank data loaded through NLTK, builds SRL-based QA examples, evaluates a legacy baseline, adds role-aware improvements through the RAISE framework, and finally creates a standalone SRL + RAG Streamlit demo.

The key strength of the project is explainability. The final system can retrieve evidence, select an answer using semantic roles, and show a graph-based reasoning path.

So the final project story is:

```text
Semantic roles make QA more structured.
Retrieval makes the evidence accessible.
Graph reasoning makes the answer explainable.
```

Thank you.

## 12. Very Short 1-Minute Version

Good morning professor.

My project is an explainable question answering system using Semantic Role Labeling and RAG. The motivation is that normal QA systems often return an answer without explaining why. I used PropBank data through NLTK to get predicate-argument structures such as `ARG0`, `ARG1`, `ARGM-LOC`, and `ARGM-TMP`.

The project has three main parts: the legacy baseline in `srl_qa_project`, the improved RAISE-SRL-QA framework in `srlqa`, and the final Streamlit SRL + RAG demo in `srl_rag_demo`.

In the final demo, the system retrieves SRL-structured evidence, selects the answer using role-aware logic, and displays a semantic graph showing the reasoning path. For example, for the question `Where was the package delivered?`, the system maps `Where` to `ARGM-LOC` and answers `to the office`.

The full baseline gives `76.12%` QA token F1 and `71.33%` SRL micro F1. On a controlled 15-example RAISE seed suite, the role-aware pipeline reaches `100.00%` exact match and role accuracy. The main contribution is connecting SRL, retrieval, QA, and graph-based explainability into a functional local demo.

Thank you.

## 13. Possible Professor Questions And Answers

### Question 1: What is the main problem you solved?

I tried to solve the problem of explainability in question answering. Instead of only returning an answer, the system shows the semantic role and evidence path behind the answer.

### Question 2: Why did you use SRL?

I used SRL because it gives structured meaning. It tells us who did the action, what was affected, where it happened, when it happened, and how it happened. This makes answer selection more interpretable.

### Question 3: Why did you use PropBank?

PropBank provides predicate-argument annotations, which are directly useful for SRL. It gives real examples of roles such as `ARG0`, `ARG1`, `ARGM-LOC`, and `ARGM-TMP`.

### Question 4: What is the difference between the baseline and your improved system?

The baseline predicts answer spans from SRL-based examples. The improved system adds role-aware retrieval, PropBank frame information, verification, and graph explanation.

### Question 5: What is the role of RAG in your project?

RAG is used to retrieve relevant SRL-structured documents before answering. This means the answer is grounded in retrieved evidence instead of being selected without context.

### Question 6: What makes the system explainable?

The system is explainable because it shows the question, retrieved document, predicate, semantic role, candidate answer span, and final answer as a graph.

### Question 7: What is your best full-test metric?

The best full-test baseline metric is `76.12%` QA token F1, with `51.84%` exact match, `71.33%` SRL micro F1, and `81.63%` BIO accuracy.

### Question 8: Can you claim 100% accuracy for the whole project?

No. The `100.00%` result is only for a controlled 15-example curated RAISE seed suite. The full-test baseline metric is separate and should be reported as `51.84%` exact match and `76.12%` QA token F1.

### Question 9: What is the final demo?

The final demo is the Streamlit app in `srl_rag_demo`. It loads PropBank, builds SRL documents, retrieves evidence, answers questions, and displays graph reasoning.

### Question 10: What is the future scope?

Future work includes running larger benchmarks, adding a stronger SRL parser for arbitrary uploaded documents, improving model training, and enhancing the graph explanation.

