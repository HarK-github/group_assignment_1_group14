# Question 2: Transition-Based Dependency Parser

This directory contains the implementation, training, and evaluation of a **Transition-Based Dependency Parser** using the **Arc-Standard (Shift-Reduce)** algorithm.

---

## 1. Overview & Architecture

The parser builds a projective dependency tree incrementally over a sentence by predicting transitions between consecutive parser states.

### State Representation
The parsing state configuration $C = (\sigma, \beta, A)$ consists of:
- **Stack ($\sigma$)**: Tracks tokens currently under consideration, initialized with `ROOT` (ID `0`).
- **Buffer ($\beta$)**: Holds upcoming input tokens from the sentence in left-to-right order.
- **Arc Set ($A$)**: Accumulates predicted labeled dependency arcs of the form $(\text{head}, \text{dependent}, \text{label})$.

### Transition System (Arc-Standard)
The parser supports three atomic transitions:
1. **`SHIFT`**: Removes the front token from the buffer and pushes it onto the top of the stack.
   - *Precondition:* Buffer is not empty ($\beta \neq \emptyset$).
2. **`LEFT-ARC(label)`**: Creates a directed dependency arc $S_1 \xrightarrow{\text{label}} S_2$, where the top of the stack ($S_1$) becomes the head of the second item ($S_2$), and pops $S_2$ from the stack.
   - *Precondition:* At least two tokens on the stack ($|\sigma| \ge 2$), and $S_2 \neq \text{ROOT}$ (the artificial root cannot be a dependent).
3. **`RIGHT-ARC(label)`**: Creates a directed dependency arc $S_2 \xrightarrow{\text{label}} S_1$, where the second item ($S_2$) becomes the head of the top item ($S_1$), and pops $S_1$ from the stack.
   - *Precondition:* At least two tokens on the stack ($|\sigma| \ge 2$).

> **Oracle Invariant:** In the Arc-Standard system, once a token is popped from the stack, it can never acquire additional children. Therefore, the gold oracle only executes an arc operation when the dependent token has already gathered **all** of its gold dependents.

### Constrained Decoding
Classifiers predict raw transition distributions without inherent grammar guarantees. During inference, the parser enforces validity preconditions:
- It checks the validity of predicted actions and selects the **highest probability-ranked valid transition** using `predict_proba`.
- This ensures the decoding process never crashes on illegal moves and always terminates with a valid, single-rooted projective dependency tree.

---

## 2. Feature Representation

The model utilizes a 4-token part-of-speech (POS) contextual window to represent each parser configuration:

| Feature Name | Description | Example |
| :--- | :--- | :--- |
| `s1_pos` | POS tag of the token at the top of the stack ($S_1$) | `VERB` |
| `s2_pos` | POS tag of the second token on the stack ($S_2$) | `NOUN` |
| `b1_pos` | POS tag of the front token in the buffer ($B_1$) | `DET` |
| `b2_pos` | POS tag of the second token in the buffer ($B_2$) | `NOUN` |

These categorical dictionary features are vectorized into sparse one-hot binary indicator vectors using `sklearn.feature_extraction.DictVectorizer`.

---

## 3. Dataset & Model Training

### Dataset
The parser is trained and evaluated on the **Universal Dependencies English Web Treebank (UD_English-EWT)** in CoNLL-U format:
- **Training Set (`en_ewt-ud-train.conllu`)**: 12,543 sentences (204,585 tokens) generating over 200,000 transition training instances.
- **Development Set (`en_ewt-ud-dev.conllu`)**: 2,001 sentences (25,148 tokens).

### Model Configuration
- **Classifier:** Multi-class `LogisticRegression` (scikit-learn)
- **Optimization Solver:** `lbfgs`
- **Max Iterations:** `500`
- **Random Seed:** `42`
- **Training Wall-Clock Time:** `64.27 seconds`
- **Training Accuracy:** **`80.43%`**

---

## 4. Evaluation & Performance

The parser was evaluated on the complete development split (`en_ewt-ud-dev.conllu`), measuring attachment accuracy and parsing throughput:

$$\text{UAS} = \frac{\text{Tokens with Correct Head}}{\text{Total Evaluated Tokens}} \times 100\%$$

$$\text{LAS} = \frac{\text{Tokens with Correct Head AND Correct Label}}{\text{Total Evaluated Tokens}} \times 100\%$$

### Quantitative Results

| Metric | Score | Detail |
| :--- | :--- | :--- |
| **Unlabeled Attachment Score (UAS)** | **66.70%** | 16,773 / 25,148 tokens correct head |
| **Labeled Attachment Score (LAS)** | **56.71%** | 14,261 / 25,148 tokens correct head & label |
| **Evaluated Sentences** | **2,001** | Full UD-EWT development split |
| **Total Evaluated Tokens** | **25,148** | Words and punctuation |
| **Inference Time** | **14.20s** | **140.9 sentences/sec** throughput |

---

## 5. Qualitative Results on Example Sentences

The trained parser was tested on three required English sentences with the following predicted dependency trees:

### Sentence 1: *"The cat sat on the mat."*
```text
         The (1)  <---[  det   ]---  cat (2)
         cat (2)  <---[  root  ]---  ROOT (0)
         sat (3)  <---[  acl   ]---  cat (2)
          on (4)  <---[  case  ]---  mat (6)
         the (5)  <---[  det   ]---  mat (6)
         mat (6)  <---[  obj   ]---  sat (3)
           . (7)  <---[ punct  ]---  cat (2)
```

### Sentence 2: *"She eats a green salad."*
```text
         She (1)  <---[ nsubj  ]---  eats (2)
        eats (2)  <---[  root  ]---  ROOT (0)
           a (3)  <---[  det   ]---  salad (5)
       green (4)  <---[  amod  ]---  salad (5)
       salad (5)  <---[  obj   ]---  eats (2)
           . (6)  <---[ punct  ]---  eats (2)
```

### Sentence 3: *"I saw the man with a telescope."*
```text
           I (1)  <---[ nsubj  ]---  saw (2)
         saw (2)  <---[  root  ]---  ROOT (0)
         the (3)  <---[  det   ]---  man (4)
         man (4)  <---[  obj   ]---  saw (2)
        with (5)  <---[  case  ]---  telescope (7)
           a (6)  <---[  det   ]---  telescope (7)
   telescope (7)  <---[  nmod  ]---  man (4)
           . (8)  <---[ punct  ]---  saw (2)
```

---

## 6. Analysis & Discussion

1. **Efficiency vs. Expressiveness:**
   - Logistic Regression trained on sparse POS indicator features executes rapidly (140.9 sentences/sec inference speed) without requiring GPU acceleration.
   - However, relying solely on a 4-token POS window ($S_1, S_2, B_1, B_2$) restricts the model's capacity to resolve semantic attachment ambiguities.
2. **Prepositional Phrase (PP) Attachment:**
   - In Sentence 3 (*"I saw the man with a telescope"*), the parser attaches *"with a telescope"* to *"man"* (`nmod`) rather than *"saw"*. Resolving such structural ambiguity strictly requires lexical identity and selectional preferences between verbs and nouns.
3. **Future Improvements:**
   - **Lexical Features:** Integrating word forms and lemmas alongside POS tags.
   - **Structural Context:** Including head and child POS tags from already constructed dependency subtrees.
   - **Neural Transition Parsing:** Transitioning from linear classifiers to neural embedding-based architectures (e.g., Chen & Manning bi-LSTM or feed-forward parsers).
