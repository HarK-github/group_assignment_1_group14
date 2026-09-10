import os
import sys
import math
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Set, Optional, Any

# Ensure Question_1 and root are on sys.path if loading Q1 model
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
Q1_DIR = os.path.join(PROJECT_ROOT, "Question_1")
for p in [CURRENT_DIR, PROJECT_ROOT, Q1_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


class SmoothedNgramModels:
    """Class for Add-k smoothed N-gram language models (Bigram & Trigram)."""
    
    def __init__(self, k: float = 0.05):
        """
        Initializes the smoothed N-gram model.
        
        Args:
            k: The smoothing parameter (default: 0.05).
        """
        self.k: float = float(k)
        self.unigrams: Counter = Counter()
        self.bigrams: Counter = Counter()
        self.trigrams: Counter = Counter()
        self.vocab: Set[str] = set()
        self.total_tokens: int = 0
        
        self.BOS: str = "<BOS>"
        self.EOS: str = "<EOS>"
        self.UNK: str = "<UNK>"

    @property
    def vocab_size(self) -> int:
        """Returns the effective vocabulary size for smoothing."""
        return max(len(self.vocab), 1)

    def train(self, sentences: List[List[str]]) -> None:
        """
        Trains the bigram and trigram models on the given sentences.
        
        Args:
            sentences: List of tokenized sentences.
        """
        for sent in sentences:
            if not sent:
                continue
            clean_sent = [str(w).lower() for w in sent if str(w).strip()]
            if not clean_sent:
                continue
                
            # Padded sequence with boundary markers
            padded = [self.BOS, self.BOS] + clean_sent + [self.EOS]
            
            # 1. Unigrams
            for w in clean_sent:
                self.unigrams[w] += 1
                self.vocab.add(w)
                
            self.unigrams[self.BOS] += 2
            self.unigrams[self.EOS] += 1
            self.vocab.add(self.BOS)
            self.vocab.add(self.EOS)
            self.total_tokens += len(clean_sent)
            
            # 2. Bigrams
            for i in range(len(padded) - 1):
                self.bigrams[(padded[i], padded[i + 1])] += 1
                
            # 3. Trigrams
            for i in range(len(padded) - 2):
                self.trigrams[(padded[i], padded[i + 1], padded[i + 2])] += 1

    def get_unigram_prob(self, word: str) -> float:
        """
        Calculates smoothed unigram probability P(word).
        """
        w = str(word).lower() if word not in (self.BOS, self.EOS) else word
        c_uni = self.unigrams.get(w, 0)
        V = self.vocab_size
        return (c_uni + self.k) / (self.total_tokens + self.k * V)

    def get_bigram_prob(self, word1: str, word2: str) -> float:
        """
        Calculates the smoothed bigram probability P(word2 | word1).
        
        Formula:
            P(w2 | w1) = (Count(w1, w2) + k) / (Count(w1) + k * |V|)
        If Count(w1) == 0, backs off to smoothed unigram probability.
        """
        w1 = str(word1).lower() if word1 not in (self.BOS, self.EOS) else word1
        w2 = str(word2).lower() if word2 not in (self.BOS, self.EOS) else word2
        
        c_bi = self.bigrams.get((w1, w2), 0)
        c_uni = self.unigrams.get(w1, 0)
        V = self.vocab_size
        
        if c_uni > 0:
            return (c_bi + self.k) / (c_uni + self.k * V)
        else:
            # Backoff to smoothed unigram probability
            return self.get_unigram_prob(w2)

    def get_trigram_prob(self, word1: str, word2: str, word3: str) -> float:
        """
        Calculates the smoothed trigram probability P(word3 | word1, word2).
        
        Formula:
            P(w3 | w1, w2) = (Count(w1, w2, w3) + k) / (Count(w1, w2) + k * |V|)
        If Count(w1, w2) == 0, backs off to smoothed bigram probability P(w3 | w2).
        """
        w1 = str(word1).lower() if word1 not in (self.BOS, self.EOS) else word1
        w2 = str(word2).lower() if word2 not in (self.BOS, self.EOS) else word2
        w3 = str(word3).lower() if word3 not in (self.BOS, self.EOS) else word3
        
        c_tri = self.trigrams.get((w1, w2, w3), 0)
        c_bi = self.bigrams.get((w1, w2), 0)
        V = self.vocab_size
        
        if c_bi > 0:
            return (c_tri + self.k) / (c_bi + self.k * V)
        else:
            # Backoff to smoothed bigram probability
            return self.get_bigram_prob(w2, w3)

    def score_sentence_bigram(self, tokens: List[str]) -> Tuple[float, float]:
        """
        Computes total log-probability and perplexity for a sentence under the Bigram model.
        
        Returns:
            Tuple of (total_log_prob, perplexity).
        """
        if not tokens:
            return 0.0, 1.0
            
        clean = [str(w).lower() for w in tokens if str(w).strip()]
        if not clean:
            return 0.0, 1.0
            
        padded = [self.BOS] + clean + [self.EOS]
        total_log_prob = 0.0
        
        for i in range(len(padded) - 1):
            prob = self.get_bigram_prob(padded[i], padded[i + 1])
            prob = max(prob, 1e-30)
            total_log_prob += math.log(prob)
            
        n_words = len(clean)
        avg_log_prob = total_log_prob / n_words
        perplexity = math.exp(-avg_log_prob) if avg_log_prob > -700 else float('inf')
        return total_log_prob, perplexity

    def score_sentence_trigram(self, tokens: List[str]) -> Tuple[float, float]:
        """
        Computes total log-probability and perplexity for a sentence under the Trigram model.
        
        Returns:
            Tuple of (total_log_prob, perplexity).
        """
        if not tokens:
            return 0.0, 1.0
            
        clean = [str(w).lower() for w in tokens if str(w).strip()]
        if not clean:
            return 0.0, 1.0
            
        padded = [self.BOS, self.BOS] + clean + [self.EOS]
        total_log_prob = 0.0
        
        for i in range(2, len(padded)):
            prob = self.get_trigram_prob(padded[i - 2], padded[i - 1], padded[i])
            prob = max(prob, 1e-30)
            total_log_prob += math.log(prob)
            
        n_words = len(clean)
        avg_log_prob = total_log_prob / n_words
        perplexity = math.exp(-avg_log_prob) if avg_log_prob > -700 else float('inf')
        return total_log_prob, perplexity

    def get_sentence_log_prob(self, tokens: List[str], order: str = 'trigram') -> float:
        """Returns the total log probability of a sentence under the chosen model ('bigram' or 'trigram')."""
        if order.lower() == 'bigram':
            lp, _ = self.score_sentence_bigram(tokens)
        else:
            lp, _ = self.score_sentence_trigram(tokens)
        return lp

    def get_sentence_perplexity(self, tokens: List[str], order: str = 'trigram') -> float:
        """Returns the perplexity of a sentence under the chosen model ('bigram' or 'trigram')."""
        if order.lower() == 'bigram':
            _, ppl = self.score_sentence_bigram(tokens)
        else:
            _, ppl = self.score_sentence_trigram(tokens)
        return ppl

    def load_from_q1_lm(self, q1_lm_or_path: Any) -> None:
        """
        Reuses the pre-trained Question 1 English TrigramLanguageModel counts
        without retraining from scratch.
        """
        if isinstance(q1_lm_or_path, str):
            with open(q1_lm_or_path, 'rb') as f:
                q1_lm = pickle.load(f)
        else:
            q1_lm = q1_lm_or_path
            
        self.unigrams = Counter(getattr(q1_lm, 'unigrams', {}))
        self.bigrams = Counter(getattr(q1_lm, 'bigrams', {}))
        self.trigrams = Counter(getattr(q1_lm, 'trigrams', {}))
        self.vocab = set(getattr(q1_lm, 'vocab', set()))
        self.total_tokens = int(getattr(q1_lm, 'total_tokens', sum(self.unigrams.values())))
        self.BOS = getattr(q1_lm, 'BOS', '<BOS>')
        self.EOS = getattr(q1_lm, 'EOS', '<EOS>')
        self.vocab.add(self.BOS)
        self.vocab.add(self.EOS)

    def save(self, filepath: str) -> None:
        """Serializes the model to a pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"SmoothedNgramModels saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SmoothedNgramModels":
        """Loads a pre-trained SmoothedNgramModels from a pickle file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"SmoothedNgramModels loaded from: {filepath}")
        return model

    @classmethod
    def load_or_train(cls, models_dir: Optional[str] = None, k: float = 0.05) -> "SmoothedNgramModels":
        """
        Loads pre-trained model from models/ or Q1 English LM, falling back
        to Brown corpus training if needed.
        """
        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models"))
            
        # 1. Try dedicated Q4 N-gram checkpoint
        q4_ngram_path = os.path.join(models_dir, "q4_smoothed_ngram.pkl")
        if os.path.exists(q4_ngram_path):
            try:
                return cls.load(q4_ngram_path)
            except Exception as e:
                print(f"Notice: Could not load {q4_ngram_path}: {e}")
                
        # 2. Reuse Q1 English TrigramLanguageModel checkpoint
        q1_lm_path = os.path.join(models_dir, "q1_lm_en.pkl")
        if os.path.exists(q1_lm_path):
            try:
                model = cls(k=k)
                model.load_from_q1_lm(q1_lm_path)
                print("Successfully initialized SmoothedNgramModels from Q1 English LM.")
                return model
            except Exception as e:
                print(f"Notice: Could not adapt {q1_lm_path}: {e}")

        # 3. Fallback: Train on Brown corpus
        print("Training SmoothedNgramModels from NLTK Brown Corpus...")
        import nltk
        from nltk.corpus import brown
        nltk.download('brown', quiet=True)
        raw_sents = brown.sents()
        sents = [[w.lower() for w in s if w.isalnum()] for s in raw_sents]
        sents = [s for s in sents if s]
        
        model = cls(k=k)
        model.train(sents)
        print(f"Trained on {len(sents):,} sentences. Vocab: {len(model.vocab):,} words.")
        return model
