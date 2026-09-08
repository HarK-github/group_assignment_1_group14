# Question 2: Dependency Parser

This directory contains the implementation of a Transition-Based Dependency Parser.

## Design Choices & Architecture
The parser uses a standard **Shift-Reduce (Transition-Based)** architecture to build a dependency tree incrementally.
- **State Representation:** The parsing state is tracked using a `State` class, which maintains a `stack` (for words currently being processed) and a `buffer` (for upcoming words).
- **Transitions:** The parser supports three primary operations:
  - `SHIFT`: Moves a word from the buffer to the stack.
  - `LEFT-ARC`: Creates a dependency link where the second item on the stack is a dependent of the top item.
  - `RIGHT-ARC`: Creates a dependency link where the top item on the stack is a dependent of the second item.
- **Oracle Training:** The `get_oracle` function acts as a dynamic oracle during training, simulating the optimal sequence of actions required to reconstruct the gold-standard dependency arcs from the training data.

## Feature Set
The model relies on a lightweight, POS-tag-based feature extraction strategy. At any given state, it extracts 4 core features:
1. `stack_top`: The POS tag of the first word on the stack.
2. `stack_second`: The POS tag of the second word on the stack.
3. `buffer_first`: The POS tag of the first word in the buffer.
4. `buffer_second`: The POS tag of the second word in the buffer.

*Note: These features are fed into a machine-learning classifier (via scikit-learn) to predict the correct transition at runtime.*

## Dataset
The parser was trained and evaluated using the **Universal Dependencies English Web Treebank (UD_English-EWT)** in CoNLL-U format.

## Performance
**Final Labeled Attachment Score (LAS) on the Dev Set:** `[INSERT FINAL LAS SCORE HERE]`

*(Note: The provided Jupyter notebook demonstrates the core logic, oracle, and feature extraction, but the final model evaluation and LAS score were not present in the notebook. Please update the placeholder above with your final score).*
