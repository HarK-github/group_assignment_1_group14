# 1. Title & Meta Information
* **Course:** DSL504 - Natural Language Processing
* **Project:** Group Assignment 1 - Statistical NLP, Parsing, and Live-Typing Systems
* **Team:**
  * Kinshuk Gupta, 12341190, kinshukg@iitbhilai.ac.in
  * Harshit Kandpal, B24CS505, harshitka@iitbhilai.ac.in
  * Sujal Som, B24DS032, sujals@iitbhilai.ac.in
  * Utkarsh Chatterjee, B24CS050, utkarshchatt@iitbhilai.ac.in
  * Tanmay Jaiswal, B24DS033, tanmay@iitbhilai.ac.in
* **Repository:** https://github.com/HarK-github/group_assignment_1_group14
* **Live Deployment:** https://groupassignment1group14-yiipvtgjukqr9x9fnirabf.streamlit.app/

# 2. Executive Summary
This repository implements a multi-part statistical NLP system spanning word segmentation, POS tagging, dependency parsing, spelling correction, grammaticality analysis, and an integrated live-typing editor. The architecture is intentionally modular: Question 1 trains trigram language models and Viterbi-based segmenters for English and Spanish, then layers HMM POS tagging and a morphology-aware tagger for richer agreement modeling. Question 2 implements a transition-based Arc-Standard dependency parser trained from oracle-generated state-action pairs over CoNLL-U data. Question 3 builds a Brown-corpus spelling correction engine that contrasts exhaustive edit-distance-1 candidate generation with a symmetric-delete lookup structure. Question 4 integrates the earlier components into a Streamlit-based background editor with live segmentation, spelling, and grammar alerts, then adds sentence-level passage analysis using PCFG, trigram, and bigram scoring. The overall design emphasizes classical NLP dynamics: dynamic programming for segmentation and parsing, sparse feature classification for dependency transitions, corpus-frequency estimation for spelling, and fast probabilistic lookup for runtime grammar triggers.

# 3. Question 1: Word Segmentation and POS Tagging
## 3.1 Core Theory and Implementation Surface
The Question 1 implementation lives primarily in [Question_1/segmenter.py](Question_1/segmenter.py), with supporting explanation and experimental outputs in [Question_1/Question_1.ipynb](Question_1/Question_1.ipynb). The code defines the following major components: `DataLoader`, `GreedySegmenter`, `TrigramLanguageModel`, `TrigramDPSegmenter`, `SegmentationMetrics`, `MFTBaseline`, `HMMPosTagger`, `AlignedErrorAttributor`, and benchmark/serialization helpers. The exported artifacts are stored under [models/README.md](models/README.md) and the `models/` directory.

The theoretical segmentation model is a trigram language model coupled to dynamic programming search. `TrigramLanguageModel` estimates unigram, bigram, and trigram counts over tokenized sentences, applies add-$k$ smoothing with $k=0.05$, and backs off from trigram to bigram and then unigram probabilities when context counts are sparse. `TrigramDPSegmenter` performs segmentation over an unspaced string using Viterbi-style dynamic programming with beam pruning. The state at each character position stores partial hypotheses, previous word context, and backpointers. The implementation limits candidate spans by `max_word_len=20`, uses `beam_width=20`, and applies a strong OOV penalty of `-18.0` to single-character unknown segments. In effect, segmentation is solved as a maximum log-probability path problem under a lexicon-constrained trigram model.

The POS tagger is a first-order HMM with Viterbi decoding. `HMMPosTagger` collects tag unigram and bigram transition counts, word-tag emission counts, and the support sets needed for smoothing. Transition probabilities are computed as add-$k$ smoothed estimates over tag bigrams; emission probabilities are computed as add-$k$ smoothed word likelihoods conditioned on tag. The decoding algorithm fills a dense trellis with log-scores and backpointers, then backtracks the best tag sequence. The code uses a very small smoothing constant, `k=1e-4`, to preserve the dominant corpus statistics while avoiding zero-probability transitions and emissions.

The baseline tagger, `MFTBaseline`, assigns each token its most frequent tag from the training corpus and falls back to the global modal tag for unseen words. The segmentation baseline, `GreedySegmenter`, performs longest-match greedy tokenization by scanning forward and matching the longest vocabulary substring at each offset. These baselines are important because they isolate the value of dynamic programming and sequence modeling: greedy segmentation is fast but locally shortsighted, while MFT tagging ignores context entirely.

## 3.2 Data and Language Coverage
The notebook and code support English and Spanish. English uses the Brown corpus with universal tags; Spanish uses the UD Spanish-GSD corpus, cached through the network when absent. The notebook output reports Brown loading as 45,727 train sentences and 11,432 test sentences, while Spanish UD-GSD is split into 14,186 train, 1,400 dev, and 427 test sentences. The extracted vocabularies reported in the notebook are 44,145 English words and 41,638 Spanish words. English POS tagging uses 11 tags, and the standard Spanish branch uses 16 tags. These counts are confirmed in the notebook outputs and align with the model registry in [models/README.md](models/README.md).

The morphologically richer Spanish branch extends the tag inventory to a compound representation that includes gender and number, producing 85 distinct morphology-aware tags. In the implementation, this is achieved by combining the UPOS tag with `Gender` and `Number` features from the UD annotation. The purpose is to force the tagger to model agreement phenomena explicitly rather than collapsing all morphosyntactic distinctions into a coarse POS label.

## 3.3 Evaluation Design and Reported Results
The notebook evaluates segmentation on 100 English and 100 Spanish test sentences. The segmentation metrics are boundary-based precision, recall, F1, and exact match rate. The notebook output reports the following results:

| Setting | Precision | Recall | F1 | Exact Match |
|---|---:|---:|---:|---:|
| English Greedy | 0.7420 | 0.7258 | 0.7292 | 15.0% |
| English DP | 0.9388 | 0.9272 | 0.9311 | 58.0% |
| Spanish Greedy | 0.6980 | 0.6480 | 0.6658 | 4.0% |
| Spanish DP | 0.9085 | 0.8902 | 0.8961 | 17.0% |

The POS results reported in the notebook are:

| Setting | MFT Accuracy | HMM Accuracy |
|---|---:|---:|
| English | 91.67% | 92.97% |
| Spanish | 87.44% | 89.12% |

The end-to-end segmentation-plus-tagging accuracies reported in the notebook are 63.85% versus 84.11% for English and 53.42% versus 75.13% for Spanish, reflecting the benefit of dynamic programming segmentation before tagging. The aligned error attribution analysis states that segmentation accounts for 72.25% of English DP errors and 80.34% of Spanish DP errors. This is a critical result because it shows that even after improving exact token boundaries, the dominant residual failure mode remains segmentation, not POS label assignment.

## 3.4 Confusion Matrices and Morphology Analysis
The notebook prints full ASCII confusion matrices for English and Spanish HMM POS tagging. The English matrix highlights frequent confusions among `NOUN`, `VERB`, `DET`, `ADP`, `PRON`, `ADV`, `ADJ`, and `PRT`, with the strongest diagonal mass on `NOUN` and `VERB`. The Spanish matrix similarly shows strong diagonal performance for `NOUN`, `ADP`, `DET`, `VERB`, `PROPN`, `ADJ`, `ADV`, and `PRON`, with proper nouns (`PROPN`) being the most error-prone among high-frequency categories. These matrices indicate that HMM tagging is substantially more reliable than naive memorization, but still vulnerable to coarse tag ambiguity and label sparsity.

The morphology-aware Spanish section prints agreement probabilities that demonstrate the benefit of encoding gender and number. The notebook output reports:

* $P(ADJ\text{-}Fem\text{-}Sing \mid NOUN\text{-}Fem\text{-}Sing) = 0.093221$
* $P(ADJ\text{-}Masc\text{-}Sing \mid NOUN\text{-}Fem\text{-}Sing) = 0.002897$
* Agreement ratio: 32.17x
* $P(ADJ\text{-}Masc\text{-}Sing \mid NOUN\text{-}Masc\text{-}Sing) = 0.091070$
* $P(ADJ\text{-}Fem\text{-}Sing \mid NOUN\text{-}Masc\text{-}Sing) = 0.000820$
* Agreement ratio: 111.10x

These numbers provide direct empirical evidence that morphology-aware tagging captures agreement constraints that a standard universal tagset would flatten.

## 3.5 Hardcoded Settings and Experimental Notes
The code and notebook hardcode several important parameters: 80/20 Brown splitting, `k=0.05` for trigram smoothing, `beam_width=20`, `max_word_len=20`, `k=1e-4` for HMM smoothing, and evaluation on the first 100 test sentences per language. The Spanish data loader caches UD-GSD files under `Question_1/ud_cache` and fetches them from GitHub if missing. The notebook also injects a demo word, `despejado`, into the training corpus with count 10 for demonstration purposes. The saved notebook output reports 407,165 transitions, 82 action classes, and 80.43% training accuracy for the HMM-related training pipeline.

## 3.6 Model Artifacts
The serialized Q1 artifacts listed in [models/README.md](models/README.md) are:

* `q1_lm_en.pkl`, `q1_dp_seg_en.pkl`, `q1_greedy_seg_en.pkl`, `q1_mft_en.pkl`, `q1_hmm_en.pkl`
* `q1_lm_es.pkl`, `q1_dp_seg_es.pkl`, `q1_greedy_seg_es.pkl`, `q1_mft_es.pkl`, `q1_hmm_es.pkl`, `q1_hmm_es_morpho.pkl`

These artifacts are exported by [models/export_models.py](models/export_models.py) and used directly by Question 4 to avoid retraining at runtime.

# 4. Question 2: Transition-Based Dependency Parser
## 4.1 Parsing Formalism
Question 2 implements an Arc-Standard transition-based dependency parser over Universal Dependencies English-EWT. The parser maintains a configuration $C = (\sigma, \beta, A)$ consisting of a stack, a buffer, and a set of dependency arcs. The three legal transitions are `SHIFT`, `LEFT-ARC(label)`, and `RIGHT-ARC(label)`. The oracle constraints are standard for Arc-Standard parsing: a token may only be popped after all of its gold dependents have already been attached. This guarantees projective tree construction during oracle simulation.

The notebook source defines a `State` class with `stack`, `buffer`, `arcs`, and `id_to_word` fields. The `id_to_word` mapping provides O(1) lookups for word forms and UPOS tags from token IDs. The `parsed_children(node)` helper derives the set of already-attached dependents for a node from the current arc set.

## 4.2 CoNLL-U Processing and Oracle Simulation
The notebook’s `parse_conllu(source)` function reads CoNLL-U formatted data line-by-line, ignores comments, skips multi-word tokens and empty nodes, and constructs a sentence as a list of dictionaries containing `id`, `form`, `upos`, `head`, and `deprel`. The training and development sets are pulled from the UD English-EWT repository via raw GitHub URLs.

The oracle function, `get_oracle(state, gold_heads, gold_labels, gold_children)`, selects the correct transition according to Arc-Standard legality and gold dependency completion. The logic is:

* If the second item on the stack is not ROOT, and the gold head of that second item is the top item, and the second item has already collected all its gold children, return `LEFT-ARC:<label>`.
* Else, if the gold head of the top item is the second item, and the top item has already collected all its gold children, return `RIGHT-ARC:<label>`.
* Otherwise, if the buffer is non-empty, return `SHIFT`.
* Otherwise, terminate with `None`.

This is a canonical dynamic oracle-style training simulation for transition parsing: the parser only learns from states that arise during projective gold derivations.

## 4.3 Feature Extraction and Model Training
The parser uses a highly compact feature set consisting of exactly four POS tags from the current configuration: `stack_top`, `stack_second`, `buffer_first`, and `buffer_second`. These are extracted from the top two stack positions and the first two buffer positions, with `NULL` placeholders when the stack or buffer is too short. The notebook explicitly states that the feature vector is built from this POS window and vectorized with `DictVectorizer` into sparse binary indicators.

Model training uses multi-class Logistic Regression with the `lbfgs` solver, `max_iter=500`, and `random_state=42`. The notebook output reports a training matrix with 407,165 transition examples and 82 unique action classes. The classifier is trained on the oracle-generated feature-action pairs produced by `generate_training_data(sentences)`, which applies the oracle, records the feature snapshot, stores the transition label, and then mutates the parser state by applying the transition.

## 4.4 Parser Inference Loop and Validity Constraints
The parsing function, `parse_sentence(sentence, model, vectorizer)`, is designed to decode a sentence by repeatedly extracting features from the current state, scoring all candidate transitions with `predict_proba`, filtering out invalid moves, and selecting the highest-probability valid action. This constrained decoding step is important: a plain classifier can emit impossible actions, so the parser must enforce transition preconditions to avoid illegal stack-buffer states. The notebook’s evaluation section indicates that this parsing loop is used both for development-set evaluation and for the three required example sentences.

## 4.5 Reported Results on UD English-EWT
The notebook output reports the following evaluation numbers:

* Training set size: 12,544 sentences in the notebook output, though the README text states 12,543. This discrepancy should be noted in any final report.
* Development set size: 2,001 sentences
* Token count: 25,148
* Training wall-clock time: 64.27 seconds
* Training accuracy: 80.43%
* Evaluation time: 14.20 seconds
* Throughput: 140.9 sentences/sec
* UAS: 66.70%
* LAS: 56.71%

The parser is then demonstrated on three sentences: “The cat sat on the mat.”, “She eats a green salad.”, and “I saw the man with a telescope.” The example outputs are consistent with standard dependency structures and expose the ambiguity of prepositional phrase attachment, especially in the third sentence where `with a telescope` attaches to `man` as `nmod` rather than `saw`.

## 4.6 Interpretation
The parser demonstrates the usual trade-off of sparse feature transition systems: the model is computationally efficient and interpretable, but the POS-only feature window leaves limited room for lexical disambiguation. This is particularly visible in PP attachment cases. A stronger parser would require lexical forms, lemmas, or neural contextual embeddings, but the current implementation is fully classical and remains appropriate for a lightweight academic baseline.

# 5. Question 3: Efficient Spelling Corrector
## 5.1 Corpus Statistics and Language Modeling
The spelling corrector is implemented in [Question_3/models.py](Question_3/models.py) and [Question_3/corrector.py](Question_3/corrector.py), with evaluation and benchmarking in [Question_3/evaluate.py](Question_3/evaluate.py) and CLI interaction in [Question_3/cli.py](Question_3/cli.py). The model is trained on the Brown corpus filtered to alphabetic lowercase words. `SpellingModels` builds:

* a unigram frequency table,
* a bigram frequency table over the full Brown token stream,
* a vocabulary set,
* and a symmetric-delete dictionary that maps every one-character deletion of every vocabulary word back to the original vocabulary entries.

The unigram model supplies $P(w)$ for non-word ranking, while the add-1 smoothed bigram model supplies $P(w_i\mid w_{i-1})$ for contextual scoring of real-word corrections.

## 5.2 Method A vs. Method B
Method A is standard edit-distance-1 candidate generation. Given a word, it enumerates deletions, transpositions, replacements, and insertions over the alphabet, then filters candidates to those present in the vocabulary. This approach is exhaustive and conceptually straightforward, but runtime grows with the number of generated strings.

Method B is symmetric delete. At training time, every vocabulary word is preprocessed into all possible one-character deletions and stored in a lookup dictionary. At runtime, the misspelled word is deleted once in all possible positions, and the resulting deletion patterns are used as dictionary keys. Candidate retrieval therefore relies on hash-map lookups rather than runtime combinatorial string generation. This is the central latency advantage of the system and explains why Method B is dramatically faster than Method A in both the README and the evaluation script.

## 5.3 Non-Word and Real-Word Correction Logic
`SpellingCorrector.correct_non_word(word, use_method='B')` returns the word unchanged if it is already in the vocabulary. Otherwise it retrieves candidate corrections using Method A or Method B, then chooses the candidate with the highest unigram probability. This is a pure dictionary/frequency-based correction path.

`SpellingCorrector.correct_real_word(phrase_tokens, target_idx, use_method='B')` checks whether an in-vocabulary word is wrong in context. It builds a candidate set for the target word, adds the original word back into consideration, and scores candidates using the previous token only. If a candidate’s probability is greater than 1.5 times the original word’s contextual probability, the candidate replaces the original. The method is therefore a local bigram-based real-word corrector with conservative thresholding to avoid over-correction.

## 5.4 Evaluation Protocol and Reported Metrics
The evaluation script, [Question_3/evaluate.py](Question_3/evaluate.py), samples 10% of Brown sentences, removes short sentences, and injects a single edit-distance mutation into one target word per sentence. It separately constructs non-word and real-word test sets, then caps each set at 500 examples for timely evaluation. The script reports the following metrics in the README and evaluation narrative:

* Non-Word Correction Accuracy: approximately 52.60%
* Real-Word Correction Accuracy: approximately 67.91%

The script also benchmarks exactly 1,000 non-word errors for the Speed Demon test, padding with additional mutated Brown vocabulary words if needed. The reported benchmark times are:

| Method | Time (seconds) |
|---|---:|
| Method A (Standard Edit Distance) | 0.0529 |
| Method B (Symmetric Delete) | 0.0046 |

This implies an approximate 11.5x speedup for Method B. The benchmark conclusion is algorithmic rather than empirical noise: Method A spends time generating and filtering many candidate strings at runtime, whereas Method B amortizes the expensive search structure into precomputed deletions and then performs only O(1)-style lookups.

## 5.5 CLI Behavior
`Question_3/cli.py` presents a continuous terminal spelling interface. It loads `SpellingModels`, instantiates `SpellingCorrector`, and processes text token-by-token. In the sample README flow, a sentence such as `I hav a good feeling about this.` yields `I had a good feeling about this.` with a measured latency of 1.55 ms in the printed example.

# 6. Question 4: Integrated Background Editor
## 6.1 System Architecture
Question 4 is the integrated runtime layer that combines the earlier statistical NLP components into a live, interactive editor. The main entrypoint is [Question_4/app.py](Question_4/app.py), which launches a Streamlit application with a typing panel, alert console, and live latency metrics. The editor pipeline itself is implemented in [Question_4/part1_live_editor.py](Question_4/part1_live_editor.py).

The architecture is:

1. A `PassageSampler` draws a contiguous paragraph from Brown.
2. A `MergeGenerator` simulates fast typing by occasionally deleting spaces between words.
3. `EditorPipeline.process_token()` consumes each typed token, runs segmentation and spelling checks immediately, and every `trigger_interval=5` words runs trigger-based grammar and real-word checks.
4. The Streamlit UI streams tokens with `time.sleep(0.25)` to mimic live typing and renders alerts in color-coded boxes.

The merge probability is set to `merge_prob=0.08`, which is a deliberately conservative simulation of missed spacebar strokes during rapid typing. The comment in the code explicitly frames this as a plausible fast-typing error model.

## 6.2 Per-Token and Trigger-Level Alerting
The live editor produces three alert types:

* `SEGMENT-ALERT` when a token appears to contain merged words and the DP segmenter can split it into valid vocabulary words,
* `SPELL-ALERT` when an out-of-vocabulary token is corrected via the spelling engine,
* `GRAMMAR-ALERT` when the recent context window looks implausible under trigram scoring or when a contextual real-word error is detected.

The per-token pipeline checks segmentation first, then spelling, and stores the corrected token for future context. Trigger checks are interval-based: every five processed tokens, the pipeline examines the most recent context window, computes a trigram log probability, and optionally runs real-word correction on the latest token. This separation is important because token-level corrections are latency-sensitive, while trigger-level grammar checks can amortize more expensive context processing over a short window.

## 6.3 PCFG Constituency Parsing and Tagset Reconciliation
The PCFG logic is implemented in [Question_4/part2_pcfg.py](Question_4/part2_pcfg.py). `train_pcfg()` induces a PCFG from Penn Treebank trees after converting them to Chomsky Normal Form with right factoring and limited horizontal Markovization (`horzMarkov=2`, `vertMarkov=0`). The parser then performs Viterbi CKY over binary and lexical rules, applies unary closure repeatedly, and reconstructs the best parse tree from backpointers.

The tagset reconciliation strategy is explicit and practical. The module defines a `BROWN_TO_PTB` mapping that converts Brown-style/HMM tags to Penn Treebank tags for adjectives, nouns, verbs, modals, adverbs, pronouns, determiners, conjunctions, prepositions, particles, possessives, interjections, cardinal numbers, foreign words, existential `EX`, wh-words, and punctuation. The `reconcile_tags()` function prefers the PCFG tag whenever it is available because the grammar itself is trained on PTB labels; the HMM tag is only used as fallback after conversion. This ensures consistency between the sequence tagger and the constituency grammar.

## 6.4 N-gram Scoring and Decision Rule
The sentence-level analysis code in [Question_4/part4_analysis.py](Question_4/part4_analysis.py) provides `score_sentence_pcfg`, `score_sentence_bigram`, `score_sentence_trigram`, and `apply_decision_rule`. The analysis layer can load Q1 and Q3 artifacts automatically from `models/`, or induce a compact treebank PCFG fallback from 100 parsed Treebank sentences if no explicit PCFG is provided.

The decision rule is hierarchical:

* Tier 1: Use PCFG if the sentence parses and the normalized PCFG log probability is at least `-15.0` nats/token.
* Tier 2: Otherwise evaluate the trigram LM. If the normalized trigram score is at least `-8.8`, classify as grammatical; if it is between `-11.0` and `-8.8`, classify as questionable; if it is below `-11.0`, fall through.
* Tier 3: Use the bigram LM. If the normalized bigram score is at least `-8.5`, classify as grammatical; otherwise classify as ungrammatical.

This is a pragmatic hybrid decision scheme that prioritizes global syntactic structure when available and local n-gram evidence when structure fails or is too sparse.

## 6.5 Shared N-gram Model Status
`Question_4/part3_ngram.py` currently contains a stubbed `SmoothedNgramModels` class with `NotImplementedError` in its constructor and methods. The analysis layer therefore falls back to Q1’s trained trigram language model or to hardcoded add-$k$ constants when necessary. This is an important repository caveat: the integrated editor architecture is fully designed around shared n-gram abstractions, but the dedicated Part 3 implementation is incomplete.

## 6.6 Speed Demon Benchmark
The benchmark in [Question_4/part5_benchmark.py](Question_4/part5_benchmark.py) measures exactly 1,000 Brown-derived corrupted words. Each sample is created by either:

* applying a random edit-distance-1 mutation, or
* forcing a space-deletion merge from adjacent Brown tokens.

The benchmark times two isolated paths:

* Pipeline A: Q1 segmentation check + Q3 spelling check
* Pipeline B: Q4 n-gram grammar/perplexity trigger check

The benchmark output obtained in this repository environment is:

| Pipeline | Total Latency | Avg. Per Word |
|---|---:|---:|
| A: Segmentation + Spelling | 0.081886 s | 0.0819 ms |
| B: Grammar Trigger | 0.002176 s | 0.0022 ms |

The measured delta is 0.0797 ms/word, and the total speedup is 37.63x. This confirms the intended architectural claim: segmentation plus candidate generation is much more expensive than hash-map-based n-gram scoring, yet both remain comfortably below a 16 ms interactive keystroke budget in this environment.

## 6.7 Streamlit Runtime Behavior
The Streamlit app caches the pipeline via `@st.cache_resource`, resets the simulation state on demand, and updates both typed text and alert buffers incrementally. The sidebar shows average token-check and trigger-check latencies from the pipeline’s internal timing lists. The main panel renders the simulated document, accepts manual input, and appends alerts to the top of the console. The runtime loop is intentionally simple and transparent so that the assignment can be demonstrated interactively without introducing neural or asynchronous complexity.

# 7. Local Execution & Reproducibility Guide
The following commands are copy-pasteable from a fresh clone of the repository.

## 7.1 Clone and Environment Setup
```bash
git clone https://github.com/HarK-github/group_assignment_1_group14.git
cd group_assignment_1_group14
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 7.2 Question 1 Execution
Question 1 is notebook-driven rather than packaged as a single CLI script. Open and run the notebook from the repository root:

```bash
jupyter notebook Question_1/Question_1.ipynb
```

If you prefer JupyterLab:

```bash
jupyter lab Question_1/Question_1.ipynb
```

The notebook trains the segmentation and tagging models, prints the evaluation tables, and exports artifacts into `models/`.

## 7.3 Question 2 Execution
Question 2 is also notebook-driven:

```bash
jupyter notebook Question_2/Question_2.ipynb
```

or:

```bash
jupyter lab Question_2/Question_2.ipynb
```

The notebook downloads UD English-EWT, builds the oracle-generated transition dataset, trains the logistic regression classifier, and evaluates UAS/LAS on the dev split.

## 7.4 Question 3 Execution
Run the interactive spelling CLI:

```bash
python3 Question_3/cli.py
```

Run the evaluation and Speed Demon benchmark:

```bash
python3 Question_3/evaluate.py
```

## 7.5 Question 4 Execution
Run the integrated Streamlit background editor:

```bash
streamlit run Question_4/app.py
```

Run the Q4 Speed Demon benchmark:

```bash
python3 Question_4/part5_benchmark.py
```

## 7.6 Model Regeneration
Rebuild the serialized model registry if needed:

```bash
python3 models/export_models.py
```

## 7.7 Reproducibility Notes
* The repository depends on NLTK corpora and may download Brown, Treebank, or UD data on first use.
* Question 4 Part 3 is currently a stub; the integrated editor falls back to Q1 and Q3 artifacts for n-gram-related scoring.
* Question 2 pulls UD English-EWT from GitHub URLs, so initial execution requires network access or cached data.
* Question 3’s benchmark and evaluation are stochastic unless the random seed is fixed externally.
