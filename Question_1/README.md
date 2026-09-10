# Question 1: Word Segmentation & Morphology-Aware POS Tagging

This directory contains the complete implementation, training, baseline comparisons, and empirical evaluation of an end-to-end **Word Segmentation and Morphology-Aware Part-of-Speech (POS) Tagging Pipeline** across two distinct typological languages: **English** (Brown Corpus) and a morphologically rich language, **Spanish** (Universal Dependencies UD Spanish-GSD).

---

## 1. Overview & Pipeline Architecture

In writing systems that lack overt whitespace delimiters (such as Chinese or Japanese), or in speech transcription, noisy text, and concatenated social media strings (e.g., hashtags, URLs), tokenization is non-trivial. This system processes raw, unspaced continuous character strings, recovers word boundaries via statistical language modeling and dynamic programming, and assigns morphosyntactic tags using a Hidden Markov Model.

```
Raw Unspaced String
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              Part 1: Word Segmentation                  │
│   Trigram LM + Viterbi Dynamic Programming (Beam Search)│
│            [Baseline: Greedy Longest-Match]             │
└─────────────────────────────────────────────────────────┘
        │
        ▼  list[str] (Recovered Word Tokens)
┌─────────────────────────────────────────────────────────┐
│            Parts 2 & 3: POS Tagging                     │
│         First-Order HMM with Viterbi Trellis            │
│  - English: 11 Universal Tags                           │
│  - Spanish Standard: 16 Universal Tags                  │
│  - Spanish Morpho-Aware: 85 Compound Tags               │
│            [Baseline: Most-Frequent-Tag (MFT)]          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
List of (Word, Tag) Tuples & Aligned Error Attribution
```

---

## 2. Theoretical Architecture & Formulations

### 2.1 Word Segmentation: Trigram Language Model + Dynamic Programming

Word segmentation is modeled as finding the token sequence $\hat{W} = (w_1, w_2, \dots, w_M)$ that maximizes the sentence log-probability under a Trigram Language Model:

$$\hat{W} = \operatorname*{arg\,max}_{w_1 \dots w_M} \sum_{i=1}^{M+1} \log P(w_i \mid w_{i-2}, w_{i-1})$$

where $w_{-1} = w_0 = \langle\text{BOS}\rangle$ and $w_{M+1} = \langle\text{EOS}\rangle$.

#### Laplace Smoothing with Hierarchical Backoff
To avoid zero-probability bottlenecks on unseen $n$-grams while rewarding attested collocations, the language model applies **Laplace (add-$k$) smoothing with backoff** ($k = 0.05$):

1. **Trigram Probability:**
   $$P_{\text{tri}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i) + k}{C(w_{i-2}, w_{i-1}) + k \cdot |V|}$$
   where $|V|$ is the vocabulary size.
2. **Backoff to Bigram:** If $C(w_{i-2}, w_{i-1}) = 0$, back off to bigram with a context penalty:
   $$P_{\text{bi}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k \cdot |V|} \implies \log P_{\text{bi}} - 1.0$$
3. **Backoff to Unigram:** If $C(w_{i-1}) = 0$, back off to unigram:
   $$P_{\text{uni}}(w_i) = \frac{C(w_i) + k}{N + k \cdot |V|} \implies \log P_{\text{uni}} - 2.0$$

#### Viterbi Dynamic Programming with Beam Search
For an unspaced input string $S$ of length $N$:
- **Trellis State:** Let $\text{dp}[j]$ store hypotheses ending at character position $j \in [1, N]$.
- **Recurrence Relation:** For every candidate word $w = S[i:j]$ where $\max(0, j - \text{max\_word\_len}) \le i < j$:
  $$\text{Score}(j, w_{\text{prev}}, w) = \max_{w_{\text{prev2}}} \left[ \text{Score}(i, w_{\text{prev2}}, w_{\text{prev}}) + \log P(w \mid w_{\text{prev2}}, w_{\text{prev}}) \right]$$
- **Beam Search Pruning:** To maintain $O(N \cdot L \cdot B)$ runtime complexity, only the top $B = 20$ highest-scoring hypotheses are retained at each character index $j$.
- **Out-of-Vocabulary (OOV) Fallback:** Single-character transitions are allowed with an OOV penalty ($\text{pen} = -18.0$) to guarantee the search path never deadlocks on unknown words.
- **Backtracking:** The optimal token sequence is recovered by tracing backpointers from $\text{dp}[N][0]$ to index $0$.

#### Segmentation Baseline: Greedy Longest-Match (`GreedySegmenter`)
At position $i$, the baseline selects the longest substring $S[i : i+l]$ present in the training vocabulary ($l \le 20$). If no match exists, it consumes a single character $S[i]$.
- *Failure Modes:* Susceptible to "false prefixes" (e.g., slicing *"together"* into *"to" + "get" + "her"* or *"standby"* into *"stand" + "by"*). Decisions are myopic and irreversible.

---

### 2.2 Part-of-Speech Tagging: First-Order HMM with Viterbi Decoding

The POS tagger models the joint sequence probability of words $W = (w_1, \dots, w_N)$ and tags $T = (t_1, \dots, t_N)$ under the Markov assumption:

$$\hat{T} = \operatorname*{arg\,max}_{t_1 \dots t_N} \prod_{i=1}^N P(w_i \mid t_i) P(t_i \mid t_{i-1})$$

#### Parameter Estimation (with Laplace Smoothing, $k = 10^{-4}$)
1. **Transition Probabilities:**
   $$P(t_i \mid t_{i-1}) = \frac{C(t_{i-1}, t_i) + k}{C(t_{i-1}) + k \cdot |T|}$$
2. **Emission Probabilities:**
   $$P(w_i \mid t_i) = \frac{C(t_i, w_i) + k}{C(t_i) + k \cdot (|V| + 1)}$$

#### Viterbi Trellis Algorithm
- **Initialization ($i = 0$):**
  $$V[0, t] = \log P(t \mid \langle\text{START}\rangle) + \log P(w_0 \mid t)$$
- **Induction ($i = 1 \dots N-1$):**
  $$V[i, t] = \max_{t'} \left( V[i-1, t'] + \log P(t \mid t') \right) + \log P(w_i \mid t)$$
  $$\text{Backpointer}[i, t] = \operatorname*{arg\,max}_{t'} \left( V[i-1, t'] + \log P(t \mid t') \right)$$
- **Termination & Backtracking:** Reconstructs the globally optimal tag sequence in $O(N \cdot |T|^2)$ time.

#### Tagging Baseline: Most-Frequent-Tag (`MFTBaseline`)
Assigns each word its unigram mode tag $\hat{t}_i = \operatorname*{arg\,max}_t C(w_i, t)$ observed in the training corpus, falling back to `'NOUN'` for unseen words.
- *Failure Modes:* Completely ignores syntactic context, failing on words with POS ambiguity (e.g., *"run"* as NOUN vs. VERB).

---

### 2.3 Morphology-Aware Tagging Extension (Spanish UD-GSD)

#### Linguistic Motivation
While English has minimal inflectional agreement, **Spanish exhibits rich grammatical concord**:
- Determiners, nouns, and adjectives must agree in **Gender** (`Masc`, `Fem`) and **Number** (`Sing`, `Plur`).
- Example: *"la casa roja"* $\to$ `DET[Fem,Sing] NOUN[Fem,Sing] ADJ[Fem,Sing]`.

#### Compound Morphological Tagset
We expand standard Universal POS tags into compound morphological representations:
$$\text{Compound Tag} = \text{UPOS} - \text{Gender} - \text{Number}$$
Examples: `NOUN-Fem-Sing`, `ADJ-Fem-Sing`, `DET-Masc-Plur`, `VERB-None-Sing`. The tag space expands from **16 UPOS tags** to **85 distinct morphological tags**.

#### Empirical Proof of Learned Grammatical Agreement
By expanding the tag state space, the HMM transition matrix directly learns syntactic concord constraints from data:

| Transition Pair | Transition Condition | Transition Probability | Empirical Finding |
| :--- | :--- | :---: | :--- |
| $P(\text{ADJ-Fem-Sing} \mid \text{NOUN-Fem-Sing})$ | **Agreement (Fem $\to$ Fem)** | **0.093221** | **32.17x more probable** than disagreement |
| $P(\text{ADJ-Masc-Sing} \mid \text{NOUN-Fem-Sing})$ | Disagreement (Fem $\to$ Masc) | 0.002897 | Strongly penalized by model |
| $P(\text{ADJ-Masc-Sing} \mid \text{NOUN-Masc-Sing})$ | **Agreement (Masc $\to$ Masc)** | **0.091070** | **111.10x more probable** than disagreement |
| $P(\text{ADJ-Fem-Sing} \mid \text{NOUN-Masc-Sing})$ | Disagreement (Masc $\to$ Fem) | 0.000820 | Strongly penalized by model |

---

### 2.4 Aligned Character-Span Error Attribution

#### The Token Count Mismatch Problem
When a segmenter makes an error, the count of predicted tokens frequently differs from gold tokens:
- Gold: `["the", "quick", "brown", "fox"]` (4 tokens)
- Predicted: `["thequick", "brown", "fox"]` (3 tokens)

A naive evaluation using `zip(predicted_tokens, gold_tokens)` silently truncates the stream, causing severe downstream alignment corruption.

#### Solution: Span-Based Attribution (`AlignedErrorAttributor`)
Each token is mapped to its exact half-open character span $[start, end)$ over the unspaced string:
1. **Segmentation-Induced Error:** A gold token whose character span $[start, end)$ was *not* produced by the segmenter. The POS tagger never had the opportunity to evaluate the correct word.
2. **Genuine Tagging Error:** A gold token whose character span $[start, end)$ was correctly isolated, but was assigned the incorrect POS tag.
3. **Correct Token:** Both character span and POS tag match gold labels.

#### Segmentation Cut Metrics (`SegmentationMetrics`)
For token sequence $W = (w_0, \dots, w_{M-1})$, boundary cuts are character offsets:
$$\text{Boundaries} = \left\{ \sum_{m=0}^t \text{len}(w_m) \;\Big|\; 0 \le t < |W| - 1 \right\}$$
- **Boundary Precision ($P$):** $\frac{|\text{Pred} \cap \text{Gold}|}{|\text{Pred}|}$
- **Boundary Recall ($R$):** $\frac{|\text{Pred} \cap \text{Gold}|}{|\text{Gold}|}$
- **Boundary F1 ($F_1$):** $\frac{2 \cdot P \cdot R}{P + R}$
- **Sentence Exact Match Rate:** $\mathbb{I}[\text{Pred Tokens} == \text{Gold Tokens}]$

---

## 3. Datasets & Preprocessing

### 3.1 English (NLTK Brown Corpus)
- **Tagset:** Universal POS tagset (`tagset='universal'`).
- **Data Splits:** Strict 80/20 train/test split:
  - **Training Set:** 45,727 sentences (930,064 tokens, vocabulary: 44,145 unique words).
  - **Test Set:** 11,432 sentences (231,128 tokens).
- **Cleaning:** Stripping punctuation, lowercasing, and filtering empty strings.

### 3.2 Spanish (Universal Dependencies UD Spanish-GSD)
- **Format:** CoNLL-U parsed structures cached locally in `ud_cache/`.
- **Standard Splits:**
  - **Train:** 14,186 sentences (383,625 tokens, vocabulary: 41,638 unique words).
  - **Dev:** 1,400 sentences.
  - **Test:** 427 sentences (11,040 tokens).
- **Cleaning & Linguistic Handling:**
  - Multi-word contraction tokens (e.g. `del` $\to$ `de` + `el`) are unpacked without duplicated spans.
  - Standalone punctuation tokens (`PUNCT`) are removed.
  - Spanish orthography and diacritics (`á, é, í, ó, ú, ü, ñ`) are strictly preserved.
  - Morphological features (`Gender`, `Number`) are extracted from token `feats` dicts.

### 3.3 Benchmark Evaluation Sets
- **100 continuous unspaced sentences** extracted from the test splits of each language.
- Spaces are stripped to simulate raw character input, while ground-truth token boundaries and POS tags are stored for evaluation.

---

## 4. Quantitative Results & Baseline Comparisons

### 4.1 Word Segmentation Performance

| Language | Model | Boundary Precision | Boundary Recall | Boundary F1 | Exact Match Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **English** | Greedy Longest-Match (Baseline) | 0.7420 | 0.7258 | 0.7292 | 15.0% |
| **English** | **Trigram LM + DP (Our Model)** | **0.9388** | **0.9272** | **0.9311** | **58.0%** |
| *Relative Gain* | *DP vs. Baseline* | *+19.68%* | *+20.14%* | *+20.19%* | *+43.0% (3.87x)* |
| **Spanish** | Greedy Longest-Match (Baseline) | 0.6980 | 0.6480 | 0.6658 | 4.0% |
| **Spanish** | **Trigram LM + DP (Our Model)** | **0.9085** | **0.8902** | **0.8961** | **17.0%** |
| *Relative Gain* | *DP vs. Baseline* | *+21.05%* | *+24.22%* | *+23.03%* | *+13.0% (4.25x)* |

---

### 4.2 POS Tagging Performance on Gold Standard Tokens

| Language | MFT Baseline Accuracy | HMM Model Accuracy | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **English (Brown)** | 91.67% | **92.97%** | +1.30% |
| **Spanish (UD-GSD)** | 87.44% | **89.12%** | +1.68% |

---

### 4.3 End-to-End Pipeline Performance & Error-Source Attribution

| Pipeline Configuration | End-to-End Accuracy | Genuine Tagging Error Rate | Segmentation-Induced Error Rate | Error Caused by Segmentation |
| :--- | :---: | :---: | :---: | :---: |
| **English: Greedy + HMM** | 63.85% | 7.94% | 28.21% | **78.04%** |
| **English: Trigram DP + HMM** | **84.11%** | **4.41%** | **11.48%** | **72.25%** |
| **Spanish: Greedy + HMM** | 53.42% | 6.82% | 39.76% | **85.36%** |
| **Spanish: Trigram DP + HMM** | **75.13%** | **4.90%** | **19.98%** | **80.34%** |

---

### 4.4 POS Confusion Matrices

#### English Confusion Matrix (Gold Rows vs. HMM Predicted Columns)
```text
Actual \ Pred      NOUN    VERB     DET     ADP    PRON     ADV     ADJ     PRT
-------------------------------------------------------------------------------
NOUN                283       8       1       0       4       4      10       2
VERB                 10     291       1       0       0       2       3       0
DET                   0       0     196       5       3       0       0       0
ADP                   1       0       0     190       0       6       0       2
PRON                  0       0       3       0     125       0       0       0
ADV                   0       0       1       6       0      99       2       4
ADJ                   1       1       1       0       0       7      96       0
PRT                   0       0       0       6       1       0       0      58
```

#### Spanish Confusion Matrix (Gold Rows vs. HMM Predicted Columns)
```text
Actual \ Pred      NOUN     ADP     DET    VERB   PROPN     ADJ     ADV    PRON
-------------------------------------------------------------------------------
NOUN                432       0       2       6      12       8       0       0
ADP                   0     419       0       0       0       0       1       0
DET                   0       0     370       0       0       0       0       3
VERB                  4       0       1     207       1       1       1       1
PROPN                31       4       6       3     104      12       4       2
ADJ                   4       1       0       3       1     134       2       0
ADV                   2       0       0       0       0       0      83       1
PRON                  1       0       8       0       0       0       0      76
```

---

## 5. Qualitative Results on Sample Sentences

The end-to-end pipeline was evaluated on mandatory assignment test strings and compound edge cases:

### Sample 1 (Spanish Standard): *"mispadrespuedenviajar"*
- **Raw Input:** `mispadrespuedenviajar`
- **Segmented Tokens:** `['mis', 'padres', 'pueden', 'viajar']`
- **Predicted POS Tags:**
  - `mis` $\to$ `DET`
  - `padres` $\to$ `NOUN`
  - `pueden` $\to$ `AUX`
  - `viajar` $\to$ `VERB`

### Sample 2 (Spanish Standard): *"elcielodespejadoesazul"*
- **Raw Input:** `elcielodespejadoesazul`
- **Segmented Tokens:** `['el', 'cielo', 'despejado', 'es', 'azul']`
- **Predicted POS Tags:**
  - `el` $\to$ `DET`
  - `cielo` $\to$ `NOUN`
  - `despejado` $\to$ `ADJ`
  - `es` $\to$ `AUX`
  - `azul` $\to$ `ADJ`

### Sample 3 (Spanish Morphology-Aware): *"lacasarojaesgrande"*
- **Raw Input:** `lacasarojaesgrande`
- **Segmented Tokens:** `['la', 'casa', 'roja', 'es', 'grande']`
- **Predicted Compound Tags (Gender & Number Concord):**
  - `la` $\to$ `DET-Fem-Sing`
  - `casa` $\to$ `NOUN-Fem-Sing`
  - `roja` $\to$ `ADJ-Fem-Sing` *(demonstrates feminine singular adjectival concord)*
  - `es` $\to$ `AUX-None-Sing`
  - `grande` $\to$ `ADJ-None-Sing`

### Sample 4 (English Standard): *"thequickbrownfoxjumpsoverthelazydog"*
- **Raw Input:** `thequickbrownfoxjumpsoverthelazydog`
- **Segmented Tokens:** `['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']`
- **Predicted POS Tags:** `[('the', 'DET'), ('quick', 'ADJ'), ('brown', 'NOUN'), ('fox', 'NOUN'), ('jumps', 'VERB'), ('over', 'ADP'), ('the', 'DET'), ('lazy', 'ADJ'), ('dog', 'NOUN')]`

### Sample 5 (Ambiguous Compounds & Foreign Words):
- **Compound 1:** `together` $\to$ `['together']` (`ADV`) *(avoiding greedy over-segmentation into `to` + `get` + `her`)*
- **Compound 2:** `standby` $\to$ `['standby']` (`NOUN`) *(avoiding greedy split `stand` + `by`)*
- **German Long Compound:** `autobahnmeistereiverwaltungsgebaeude` $\to$ `['autobahn', 'meisterei', 'verwaltungs', 'gebaeude']`

---

## 6. Analysis & Discussion

### 1. Where English and Spanish Differ Most in Accuracy
1. **Word Segmentation Exact Match:** English achieves **58.0% exact match** compared to only **17.0% for Spanish** (a 3.4x disparity).
2. **Morphological Sparsity:** Spanish is a fusional Romance language with extensive inflectional paradigms: verbs inflect across 40+ forms (tenses, moods, aspects, persons), while nouns and adjectives inflect for both gender and number. This yields a much higher type-token ratio and vocabulary sparsity. Unseen or rare inflected forms degrade boundary confidence.
3. **Sentence Length Effect:** Sentences in UD Spanish-GSD average ~28 tokens vs. ~18 tokens in the Brown corpus. Because sentence exact match requires every single cut to be flawless ($P(\text{Exact}) \approx p_{\text{token}}^N$), longer sentences face an exponential penalty.

### 2. Agreement-Aware Tagging: Disambiguation vs. Data Sparsity
1. **Syntactic Disambiguation:** As demonstrated by the transition matrix test, the morphology-aware model strongly enforces grammatical concord ($P(\text{ADJ-Fem} \mid \text{NOUN-Fem})$ is **32.2x higher** than discordant masculine transitions). It successfully disambiguates modifiers that share lexical stems.
2. **The Sparsity Trade-off:** Expanding the tagset from 16 to 85 states expands the transition matrix from $16^2 = 256$ to $85^2 = 7,225$ cells. For rare combinations, parameter estimation variance increases. Agreement-aware tagging provides valuable structural inductive bias, but requires adequate smoothing or subword feature sharing.

### 3. Error Attribution: The Segmentation Bottleneck
Aligned character-span error attribution reveals a striking empirical finding:
- **72.25% of all downstream tagging errors in English** and **80.34% in Spanish** are caused directly by **segmentation failures**, rather than failures of the POS tagger.
- Evaluated on gold tokens, the HMM tagger achieves high accuracy (**92.97%** on English, **89.12%** on Spanish).
- However, when a segmenter merges or fractures words (e.g. `the quick` $\to$ `thequick`), the POS tagger is forced to classify an invalid out-of-vocabulary word. **Front-end segmentation is the single dominant error bottleneck in unspaced text processing.**

### 4. Improvement over Heuristic Baselines
1. **Segmentation:** Trigram DP improves boundary F1 by **+20.19% on English** (0.7292 $\to$ 0.9311) and **+23.03% on Spanish** (0.6658 $\to$ 0.8961), with Exact Match Rate leaping by **3.8x to 4.2x**.
2. **POS Tagging:** The HMM improves over the MFT baseline by +1.30% to +1.68% by capturing syntactic transitions ($P(t_i \mid t_{i-1})$) to resolve polysemous word ambiguities.
3. **End-to-End Pipeline:** End-to-end accuracy jumps from **63.85% to 84.11% on English** (+20.26%) and from **53.42% to 75.13% on Spanish** (+21.71%).

---

## 7. Project Structure

```
Question_1/
├── README.md               # Comprehensive documentation and technical report
├── Question_1.ipynb        # Complete interactive Jupyter notebook with executions
├── segmenter.py            # Modular, production-ready Python library
├── __init__.py             # Package initialization
└── ud_cache/               # Local cache for Spanish UD-GSD CoNLL-U splits
    ├── es_gsd-ud-train.conllu
    ├── es_gsd-ud-dev.conllu
    └── es_gsd-ud-test.conllu
```

### Modular Components in `segmenter.py`
- `DataLoader`: Downloads and parses the Brown corpus and UD Spanish-GSD with contraction filtering and morphological feature extraction.
- `GreedySegmenter`: Heuristic maximum forward-matching baseline.
- `TrigramLanguageModel`: N-gram model with Laplace smoothing and backoff.
- `TrigramDPSegmenter`: Viterbi dynamic programming segmenter with beam search pruning ($B=20$) and OOV penalty.
- `MFTBaseline`: Most-frequent-tag unigram baseline POS tagger.
- `HMMPosTagger`: First-order HMM POS tagger with Viterbi decoding.
- `SegmentationMetrics`: Computes boundary Precision, Recall, F1, and Exact Match.
- `AlignedErrorAttributor`: Character-span error attribution preventing `zip()` truncation.
- `run_benchmark`: Evaluation harness computing all metrics across continuous test sentences.
- `save_model` / `load_model`: Pickle serialization utilities for fast deployment.

---

## 8. Installation & Execution

### 8.1 Setup Environment
Ensure Python 3.10+ is installed with the necessary dependencies:

```bash
# From the repository root
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 8.2 Interactive Notebook
To run the full experimental pipeline with visualizations and output logs:
```bash
jupyter notebook Question_1/Question_1.ipynb
```

### 8.3 Python Library Usage
Models can be instantiated, trained, or loaded directly from pre-trained checkpoints in `models/`:

```python
import sys
import os

# Add repository root to path
sys.path.insert(0, os.path.abspath("."))

from Question_1.segmenter import load_model

# Load pre-trained models from the central registry
dp_seg_en = load_model("models/q1_dp_seg_en.pkl")
hmm_en = load_model("models/q1_hmm_en.pkl")

# Segment unspaced text
unspaced_text = "thequickbrownfoxjumpsoverthelazydog"
tokens = dp_seg_en.segment(unspaced_text)
print("Segmented Tokens:", tokens)

# Tag with HMM Viterbi decoder
tags = hmm_en.viterbi_decode(tokens)
print("Tagged Sequence:", list(zip(tokens, tags)))
```

### 8.4 Re-training & Exporting Models
To re-train all language models, segmenters, and taggers from scratch and persist them to `models/`:
```bash
python models/export_models.py
```
