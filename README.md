# Context-Aware Spelling Correction Engine

## Overview
This repository contains a highly optimized, context-aware spelling correction engine developed for the DSL504 Natural Language Processing course. The system detects and corrects both **non-word errors** (out-of-vocabulary) and **real-word errors** (in-vocabulary but contextually incorrect) using statistical language modeling derived from the NLTK Brown Corpus.

The project features a comparative architectural analysis between Standard Edit-Distance candidate generation and the highly efficient Symmetric Delete algorithm, culminating in a low-latency Continuous Terminal CLI.

## Theoretical Architecture

### 1. Language Modeling
- **Unigram Model:** Utilized for baseline frequency distribution to resolve non-word errors. Candidates are ranked based on their standalone probability $P(w)$ in the corpus.
- **Bigram Model (Context-Aware):** Utilized for real-word error correction. The system evaluates the conditional probability $P(w_i | w_{i-1})$ using Add-1 smoothing. If a candidate correction yields a significantly higher contextual probability than the original word, a replacement is triggered.

### 2. Candidate Generation Methodologies
- **Method A (Standard Edit Distance):** Computes all possible deletions, transpositions, replacements, and insertions dynamically at an edit distance of 1. While exhaustive, this approach requires calculating hundreds of variations per query.
- **Method B (Symmetric Delete):** A heavily optimized approach that maps every possible 1-character deletion of vocabulary words in a pre-computed dictionary. At runtime, the algorithm only computes deletions of the misspelled word and performs an O(1) dictionary lookup, drastically reducing latency.

## Project Structure
- `models.py`: Corpus extraction, vocabulary building, bigram/unigram probabilities, and candidate generation logic.
- `corrector.py`: The core logical routing for handling non-word vs. real-word errors based on statistical thresholds.
- `evaluate.py`: The benchmarking suite that randomly injects mutations into the corpus to calculate accuracy and test execution latency.
- `cli.py`: The Continuous Terminal UI for live, real-time spelling correction.

## Installation & Setup
To run this engine locally, ensure you have Python 3.10+ installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kinshuk18/Q3_DSL504_NLP_Group_Assignment_1.git
   cd Q3_DSL504_NLP_Group_Assignment_1
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv q3_spell_engine
   source q3_spell_engine/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install nltk
   ```

## Execution

### 1. Live Interactive CLI
To launch the real-time spelling corrector with terminal highlighting:
```bash
python3 cli.py
```

**Sample Output:**
```
Enter a sentence: I hav a good feeling about this.
Output: I had a good feeling about this.
[Latency: 1.55 ms]
```

### 2. Evaluation & Benchmarking
To run the evaluation script against 10% of the Brown Corpus and trigger the "Speed Demon" benchmark:
```bash
python3 evaluate.py
```

## Performance Metrics & Conclusion
The system was evaluated against dynamically mutated subsets of the Brown corpus.

- **Non-Word Correction Accuracy:** ~52.60%
- **Real-Word Correction Accuracy:** ~67.91%

### Speed Demon Benchmark Results (1,000 queries)
| Method | Time (seconds) |
|---|---|
| Method A (Standard Edit Distance) | 0.0529 |
| Method B (Symmetric Delete) | 0.0046 |

**Conclusion:** Method B achieves execution latency approximately 11.5x faster than Method A. This delta is fundamentally algorithmic. Method B leverages spatial memory by pre-computing variants, allowing candidate retrieval via O(1) dictionary lookups. Method A relies on CPU-bound dynamic string generation at runtime, creating a massive computational bottleneck for every token processed.