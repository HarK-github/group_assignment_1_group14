import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")

import nltk
from nltk.corpus import brown
from collections import Counter, defaultdict
import string

# Ensure the corpus is downloaded
nltk.download('brown', quiet=True)

class SpellingModels:
    def __init__(self):
        print("Initializing Corpus and Models...")
        # Part 1: Vocabulary and Frequencies (Unigram Model)
        # Convert to lowercase to ensure consistency
        self.words = [w.lower() for w in brown.words() if w.isalpha()]
        self.vocab = set(self.words)
        self.unigram_counts = Counter(self.words)
        self.total_words = sum(self.unigram_counts.values())
        
        # Part 1: Language Model (Bigram Model)
        self.bigrams = list(nltk.bigrams(self.words))
        self.bigram_counts = Counter(self.bigrams)
        
        # Part 2, Method B: Preprocessing Step for Symmetric Delete
        self.sym_spell_dict = defaultdict(list)
        self._build_symmetric_delete_dict()
        print("Initialization Complete.")

    def _build_symmetric_delete_dict(self):
        """Creates a dictionary mapping 1-char deletions to original words."""
        for word in self.vocab:
            deletions = self._get_deletions(word)
            for delete in deletions:
                self.sym_spell_dict[delete].append(word)

    def _get_deletions(self, word):
        """Helper to generate all 1-character deletions of a word."""
        return set(word[:i] + word[i+1:] for i in range(len(word)))

    def get_unigram_prob(self, word):
        return self.unigram_counts[word] / self.total_words if self.total_words > 0 else 0

    def get_bigram_prob(self, word1, word2):
        """Calculates P(word2 | word1) with basic add-1 smoothing to avoid zero probs."""
        bigram_count = self.bigram_counts[(word1, word2)] + 1
        unigram_count = self.unigram_counts[word1] + len(self.vocab)
        return bigram_count / unigram_count

    # Part 2: Method A - Standard Edit Distance 1 Generation
    def method_a_candidates(self, word):
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        
        all_edits = set(deletes + transposes + replaces + inserts)
        # Return only edits that are real words in our vocabulary
        return set(w for w in all_edits if w in self.vocab)

    # Part 2: Method B - Symmetric Delete Spelling Correction
    def method_b_candidates(self, misspelled_word):
        candidates = set()
        # Candidate Generation Step: generate 1-char deletions of the misspelled word
        deletions = self._get_deletions(misspelled_word)
        
        # Look up these variants in the pre-processed dictionary
        for delete in deletions:
            if delete in self.sym_spell_dict:
                candidates.update(self.sym_spell_dict[delete])
        
        # Also check if the word itself missing a character is in the dict
        if misspelled_word in self.sym_spell_dict:
             candidates.update(self.sym_spell_dict[misspelled_word])
             
        return candidates