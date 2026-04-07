# Model (PropQA-Net)

PropQA-Net is a multi-task BiLSTM model that predicts:

- SRL BIO tags over the context tokens
- an extractive answer span for a question about a specific PropBank role

The implementation lives in `model.py`.

## Inputs

The model consumes fixed-shape tensors created by `data_loader.SRLQADataset`:

- `context_ids`: token ids for the sentence
- `pos_ids`: POS tag ids aligned to the sentence tokens
- `predicate_flags`: 0/1 indicator marking the predicate anchor token
- `context_mask`: boolean mask for sentence padding
- `question_ids`: token ids for the question
- `question_mask`: boolean mask for question padding

During training, it also consumes supervision:

- `label_ids`: BIO SRL tag ids
- `answer_starts`, `answer_ends`: gold span boundaries

## Architecture

Conceptually:

1. Embed context tokens (word embedding shared with question), POS tags, and predicate flags.
2. Encode the context with a BiLSTM.
3. Predict SRL BIO tags from each context hidden state.
4. Encode the question with a second BiLSTM and pool to a question vector.
5. Score start and end positions for the answer span using the context states and question vector.

## Loss

Training uses a weighted combination of SRL and QA losses:

- SRL: token-level cross-entropy over BIO tags
- QA: cross-entropy for start index + cross-entropy for end index

The interpolation weight is `ProjectConfig.model.alpha`.

## Decoding

`PropQANet.predict(...)` returns a `PredictionResult` per example. Decoding uses:

- boundary probabilities from the span heads
- BIO predictions decoded into candidate argument spans
- a similarity score between each candidate span vector and the question vector

If no candidate spans decode cleanly, the model falls back to the best boundary span and assigns a majority role from the predicted BIO window.

## Limitations

- The system is extractive and token-based: it cannot produce free-form answers.
- Alignment depends on local Treebank availability: not all PropBank instances are usable.
- Predicate anchoring is deterministic in training (from PropBank) but heuristic in raw-text inference.

