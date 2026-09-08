# Centralized Model Registry (`models/`)

This directory contains serialized checkpoints (`.pkl`) for trained models from **Question 1** and **Question 3**. Centralizing these models enables sub-second loading for the Question 4 pipeline without re-training over large text corpora on every invocation.

---

## Model Inventory

### Question 1: Word Segmentation & POS Tagging

| Filename | Description | Class | Language |
|---|---|---|---|
| `q1_lm_en.pkl` | Trigram Language Model ($k=0.05$ Laplace smoothing + backoff) | `TrigramLanguageModel` | English |
| `q1_dp_seg_en.pkl` | Dynamic Programming Viterbi Word Segmenter (Beam width 20) | `TrigramDPSegmenter` | English |
| `q1_greedy_seg_en.pkl` | Longest-match greedy baseline segmenter | `GreedySegmenter` | English |
| `q1_mft_en.pkl` | Most-Frequent-Tag baseline POS tagger | `MFTBaseline` | English |
| `q1_hmm_en.pkl` | First-order HMM POS Tagger with Viterbi decoding | `HMMPosTagger` | English |
| `q1_lm_es.pkl` | Trigram Language Model ($k=0.05$ Laplace smoothing + backoff) | `TrigramLanguageModel` | Spanish |
| `q1_dp_seg_es.pkl` | Dynamic Programming Viterbi Word Segmenter (Beam width 20) | `TrigramDPSegmenter` | Spanish |
| `q1_greedy_seg_es.pkl` | Longest-match greedy baseline segmenter | `GreedySegmenter` | Spanish |
| `q1_mft_es.pkl` | Most-Frequent-Tag baseline POS tagger | `MFTBaseline` | Spanish |
| `q1_hmm_es.pkl` | Standard Universal POS Tagging HMM with Viterbi decoding | `HMMPosTagger` | Spanish |
| `q1_hmm_es_morpho.pkl` | Morphology-Aware HMM POS Tagger (capturing Gender & Number agreement) | `HMMPosTagger` | Spanish |

### Question 3: Context-Aware Spelling Correction

| Filename | Description | Class |
|---|---|---|
| `q3_spelling_models.pkl` | Precomputed Brown Corpus Unigram, Bigram, Vocab, and Symmetric Delete dictionary | `SpellingModels` |

---

## Quick Usage

### Loading Question 1 Models
```python
import sys
sys.path.append("..") # or project root

from Question_1.segmenter import load_model

# Load English segmenter and POS tagger
dp_seg_en = load_model("models/q1_dp_seg_en.pkl")
hmm_en = load_model("models/q1_hmm_en.pkl")

# Segment and tag
tokens = dp_seg_en.segment("thequickbrownfox")
tags = hmm_en.viterbi_decode(tokens)
print(list(zip(tokens, tags)))
# Output: [('the', 'DET'), ('quick', 'ADJ'), ('brown', 'NOUN'), ('fox', 'NOUN')]
```

### Loading Question 3 Spelling Engine
```python
from Question_3.models import SpellingModels
from Question_3.corrector import SpellingCorrector

# Instant loading (<0.2s) without recomputing corpus bigrams
models = SpellingModels.load("models/q3_spelling_models.pkl")
corrector = SpellingCorrector(models)

# Non-word error correction
print(corrector.correct_non_word("speling"))  # Output: 'spelling'

# Real-word error correction
sentence = ["i", "hav", "a", "good", "feeling"]
print(corrector.correct_real_word(sentence, 1))  # Output: 'had' or 'have'
```

---

## Re-generating Models
To re-train and export all models from scratch:
```bash
python models/export_models.py
```
