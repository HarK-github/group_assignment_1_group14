# Person B: POS Tagging & Morphology - Quick Start Guide

## Files Overview

| File | Purpose |
|------|---------|
| `pos_morphology_tagger.py` | Core implementation (Parts 2, 3, 4) |
| `integration_examples.py` | 6 practical examples + testing |
| `QUICK_START.md` | This guide |

---

## Installation

```bash
# Core dependencies
pip install nltk scikit-learn numpy

# For Spanish UD corpus (optional)
pip install conllu requests
```

---

## Quick Usage

### 1️⃣ Start Immediately (Before Person A Finishes)

Use mock gold-standard tokens while Person A works on segmentation:

```python
from pos_morphology_tagger import POSPipeline, DataLoader

# Load training data
train_sents, test_sents = DataLoader.load_brown_corpus()

# Train all models
pipeline = POSPipeline(language='english')
pipeline.train_baselines_and_models(train_sents)

# Test with gold tokens from corpus (no segmentation needed)
test_sent = test_sents[0]
tokens = [w for w, t in test_sent]
tags = [t for w, t in test_sent]

# Get predictions
baseline_tags = pipeline.tag_with_baseline(tokens)  # MFT baseline
hmm_tags = pipeline.tag_with_hmm(tokens)            # HMM model

print(f"Tokens:   {tokens}")
print(f"Gold:     {tags}")
print(f"Baseline: {baseline_tags}")
print(f"HMM:      {hmm_tags}")
```

---

### 2️⃣ Integration Day (After Person A Delivers)

On integration day, use Person A's segmented output directly:

```python
# Person A's output: a list of strings
person_a_segmented = ["la", "casa", "roja", "es", "grande"]

# Person B processes it:
tags = pipeline.tag_with_hmm(person_a_segmented)
print(f"Segmented: {person_a_segmented}")
print(f"Tagged:    {tags}")
```

**Interface Contract:**
- Input: `list[str]` — tokens from Person A's segmenter
- Output: `list[str]` — POS tags

---

## Implementation Details

### Part 2: HMM POS Tagger

**What it does:**
- Learns P(tag | previous_tags) and P(word | tag)
- Uses Viterbi algorithm for decoding
- Better than baseline when data is abundant

**Class:** `HMMPosTagger`

```python
hmm = HMMPosTagger(smoothing='laplace', k=1e-5)
hmm.train(train_sents)
predicted_tags = hmm.viterbi_decode(tokens)
```

**Marks:** ~15 marks (DP part 2)

---

### Part 3: Morphology-Aware Tagging

**What it does:**
- Extends POS tags with gender/number: `NOUN` → `NOUN-Fem-Sing`
- Trains HMM on compound tags
- Captures agreement patterns

**Class:** `MorphologyAwareTagger`

```python
morpho = MorphologyAwareTagger(language='spanish')
morpho.train_from_ud(conllu_data)  # UD Spanish-GSD
morpho_tags = morpho.predict(tokens)
```

**Marks:** ~15 marks (part 3)

---

### Part 4: Baselines

**MFT Baseline:**
- Most frequent tag per word
- Falls back to `NOUN` for unseen words
- Quick baseline for comparison

**Class:** `MFTBaseline`

```python
mft = MFTBaseline(fallback_tag='NOUN')
mft.train(train_sents)
baseline_tags = mft.predict(tokens)
```

**Marks:** ~6.5 marks (baseline part 4)

---

## Evaluation (~7.5 marks)

### Error Attribution: Segmentation vs. Tagging

```python
from pos_morphology_tagger import POSEvaluator

# You receive segmented tokens from Person A
predicted_tokens = person_a_segmented  # May have errors
predicted_tags = pipeline.tag_with_hmm(predicted_tokens)

# Evaluate against gold standard
gold_tokens = ["la", "casa", "roja", "es", "grande"]
gold_tags = ["DET", "NOUN", "ADJ", "AUX", "ADJ"]

# This breaks down errors:
# - Segmentation errors (token mismatch) → Person A's fault
# - Tagging errors (token match, tag mismatch) → Person B's fault
result = POSEvaluator.attribute_errors(
    predicted_tokens, gold_tokens,
    predicted_tags, gold_tags
)

print(f"Correct:              {result['correct']}")
print(f"Tagging errors:       {result['tagging_errors']}")
print(f"Segmentation errors:  {result['segmentation_errors']}")
```

### Confusion Matrix

```python
metrics = POSEvaluator.evaluate(predicted_tags, gold_tags)

accuracy = metrics['accuracy']
cm = metrics['confusion_matrix']
# Use sklearn to visualize
```

**What to report:**
- Overall accuracy
- Per-tag precision/recall
- Confusion matrix (most confused tag pairs)
- Error analysis: Did morphology help or hurt?

---

## Running Examples

### All Examples at Once
```bash
python integration_examples.py
```

### Individual Examples

```python
from integration_examples import (
    example_1_mock_data,
    example_2_integration_day,
    example_3_error_attribution,
    example_4_confusion_matrix,
    example_5_spanish_morphology,
    example_6_standalone_testing
)

# Run one or more:
example_1_mock_data()
example_2_integration_day()
# ... etc
```

---

## Mark Allocation

| Component | Marks | Status |
|-----------|-------|--------|
| **Part 2: HMM POS Tagger** | 15 | ✅ Implemented |
| **Part 3: Morphology** | 15 | ✅ Implemented |
| **Part 4: Baselines** | 6.5 | ✅ Implemented |
| **Evaluation** | 7.5 | ✅ Implemented |
| **Report** | ~8 | 📝 To write |
| **Total** | ~52 | |

---

## Datasets

### English (Brown Corpus)
```python
train_sents, test_sents = DataLoader.load_brown_corpus(split_ratio=0.8)
# 1,161 train sentences, 290 test sentences
# Tag set: Universal POS tags (12 tags)
```

### Spanish (UD Spanish-GSD)
```python
datasets = DataLoader.load_ud_spanish()
# datasets['train'], datasets['dev'], datasets['test']
# Includes morphological features (Gender, Number, etc.)
```

---

## Common Pitfalls

### 1. OOV (Out-of-Vocabulary) Words
- Handled by fallback in MFT: uses `NOUN` for unseen words
- HMM uses Laplace smoothing: `P(word|tag) = (count + k) / (tag_count + k*vocab_size)`

### 2. Segmentation Errors Propagate
- If Person A's segmenter makes errors, tagging accuracy will drop
- Use `error_attribution()` to measure impact separately

### 3. Morphology Data Sparsity
- With `NOUN-Fem-Sing` etc., tag vocabulary ~3–4x larger
- Needs more training data or heavier smoothing
- Compare with vs. without morphology in report

### 4. Viterbi Efficiency
- Current implementation is O(n × |tags|²) — fine for small tag sets
- For larger tagsets, consider beam search

---

## What to Submit (Report)

**Report sections for Person B:**

1. **Tagging Baseline:** MFT results, comparison to random baseline
2. **HMM Model:** Explain Viterbi algorithm, show accuracy gains
3. **Morphology Impact:**
   - Does adding gender/number help or hurt?
   - Show compound tag accuracy (NOUN-Fem-Sing vs. NOUN only)
4. **Error Analysis:**
   - Confusion matrix (top 5 most confused pairs)
   - Distinguish tagging errors from segmentation errors
   - Which tag types are hardest?
5. **English vs. Spanish:**
   - Compare results across languages
   - Discuss morphology availability (Spanish has it, English doesn't)

---

## Quick Sanity Checks

```python
# Test 1: MFT gives consistent output
mft = MFTBaseline()
mft.train(train_sents)
tags1 = mft.predict(["the", "dog", "run"])
tags2 = mft.predict(["the", "dog", "run"])
assert tags1 == tags2, "MFT should be deterministic"

# Test 2: HMM learns something
hmm = HMMPosTagger()
hmm.train(train_sents)
# If vocab size = 0, training failed
assert len(hmm.vocab) > 0, "HMM training failed"

# Test 3: Morphology tags are created
morpho = MorphologyAwareTagger(language='spanish')
morpho.train_from_ud(datasets['train'][:100])
assert len(morpho.morpho_tags) > 0, "Morphology training failed"
```

---

## Need Help?

- **Viterbi not decoding?** Check that tag vocabulary is built during training
- **Low accuracy?** Increase training data, reduce smoothing parameter `k`
- **OOV errors?** Ensure fallback tags are defined
- **Morphology not working?** Verify UD features are present in corpus

---

## Checklist Before Submission

- [ ] MFT baseline implemented and tested
- [ ] HMM with Viterbi decoding working
- [ ] Morphology-aware tagger trained on Spanish
- [ ] Confusion matrix and error analysis done
- [ ] Error attribution (segmentation vs. tagging) calculated
- [ ] Report written with insights
- [ ] Integration interface tested with mock segmenter
- [ ] All examples run without errors
