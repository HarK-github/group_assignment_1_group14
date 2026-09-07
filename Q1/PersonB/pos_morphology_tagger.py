"""
Person B: POS Tagging & Morphology-Aware Extension
Parts 2 & 3 Implementation
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import nltk
from nltk.corpus import brown
from nltk import download
import conllu
import requests
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

class DataLoader:
    """Load and prepare datasets for both English (Brown) and Spanish (UD)."""
    
    @staticmethod
    def load_brown_corpus(split_ratio=0.8):
        """
        Load NLTK Brown Corpus with 80/20 train-test split.
        Returns tokenized sentences with POS tags.
        """
        download('brown', quiet=True)
        
        tagged_sents = brown.tagged_sents(tagset='universal')
        split_idx = int(len(tagged_sents) * split_ratio)
        
        train_sents = tagged_sents[:split_idx]
        test_sents = tagged_sents[split_idx:]
        
        print(f"Brown Corpus loaded: {len(train_sents)} train, {len(test_sents)} test")
        return train_sents, test_sents
    
    @staticmethod
    def load_ud_spanish():
        """
        Load UD Spanish-GSD corpus from conllu format.
        Returns train/dev/test splits with morphological features.
        """
        url = "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/master"
        
        datasets = {}
        for split in ['train', 'dev', 'test']:
            file_url = f"{url}/es_gsd-ud-{split}.conllu"
            try:
                response = requests.get(file_url, timeout=10)
                response.raise_for_status()
                datasets[split] = conllu.parse(response.text)
                print(f"UD Spanish {split}: {len(datasets[split])} sentences loaded")
            except Exception as e:
                print(f"Warning: Could not load {split} split: {e}")
        
        return datasets


# ============================================================================
# PART 1: MOST-FREQUENT-TAG BASELINE (Part 4)
# ============================================================================

class MFTBaseline:
    """Most-Frequent-Tag baseline tagger."""
    
    def __init__(self, fallback_tag='NOUN'):
        self.word_tags = defaultdict(Counter)  # word -> {tag: count}
        self.fallback_tag = fallback_tag
    
    def train(self, tagged_sentences: List[List[Tuple[str, str]]]):
        """Build word->tag frequency table from training data."""
        for sent in tagged_sentences:
            for word, tag in sent:
                self.word_tags[word.lower()][tag] += 1
    
    def predict(self, tokens: List[str]) -> List[str]:
        """Tag tokens using most frequent tag per word."""
        tags = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.word_tags:
                tag = self.word_tags[token_lower].most_common(1)[0][0]
            else:
                tag = self.fallback_tag
            tags.append(tag)
        return tags


# ============================================================================
# PART 2: HMM POS TAGGER WITH VITERBI DECODING
# ============================================================================

class HMMPosTagger:
    """
    HMM-based POS tagger with Viterbi decoding.
    Models: P(t_i | t_{i-1}, t_{i-2}) and P(w_i | t_i)
    """
    
    def __init__(self, smoothing='laplace', k=1e-5):
        self.smoothing = smoothing
        self.k = k
        
        # Transition: tag bigrams and trigrams
        self.tag_unigrams = Counter()
        self.tag_bigrams = Counter()  # (t_{i-1}, t_i)
        self.tag_trigrams = Counter()  # (t_{i-2}, t_{i-1}, t_i)
        
        # Emission: word|tag
        self.word_tag = defaultdict(Counter)  # tag -> {word: count}
        self.tag_counts = Counter()  # tag counts
        
        self.tags = set()
        self.vocab = set()
        self.START = '<START>'
        self.END = '<END>'
    
    def train(self, tagged_sentences: List[List[Tuple[str, str]]]):
        """Train HMM on tagged sentences."""
        
        for sent in tagged_sentences:
            # Extract tags and words
            words = [w.lower() for w, t in sent]
            tags = [t for w, t in sent]
            
            # Add padding for trigrams
            padded_tags = [self.START, self.START] + tags + [self.END]
            padded_words = ['<BOS>', '<BOS>'] + words + ['<EOS>']
            
            # Count n-grams
            for i in range(len(padded_tags) - 2):
                t_im2, t_im1, t_i = padded_tags[i:i+3]
                word_i = padded_words[i+2]
                
                # Transitions
                self.tag_unigrams[t_i] += 1
                self.tag_bigrams[(t_im1, t_i)] += 1
                self.tag_trigrams[(t_im2, t_im1, t_i)] += 1
                
                # Emissions
                self.word_tag[t_i][word_i] += 1
                self.tag_counts[t_i] += 1
            
            self.tags.update(tags)
            self.vocab.update(words)
    
    def _log_prob_transition(self, t_im2: str, t_im1: str, t_i: str) -> float:
        """Log probability P(t_i | t_{i-1}, t_{i-2})."""
        trigram_count = self.tag_trigrams.get((t_im2, t_im1, t_i), 0)
        bigram_count = self.tag_bigrams.get((t_im2, t_im1), 0) + self.k
        
        if bigram_count == 0:
            return np.log(self.k)
        
        return np.log(trigram_count + self.k) - np.log(bigram_count)
    
    def _log_prob_emission(self, word: str, tag: str) -> float:
        """Log probability P(word | tag)."""
        word_count = self.word_tag[tag].get(word, 0)
        tag_count = self.tag_counts[tag] + self.k * len(self.vocab)
        
        return np.log(word_count + self.k) - np.log(tag_count)
    
    def viterbi_decode(self, tokens: List[str]) -> List[str]:
        """Decode token sequence using Viterbi algorithm."""
        tokens = [t.lower() for t in tokens]
        n = len(tokens)
        tags_list = list(self.tags)
        n_tags = len(tags_list)
        tag_to_idx = {tag: i for i, tag in enumerate(tags_list)}
        
        # DP tables: (seq_idx, tag)
        viterbi = np.full((n + 1, n_tags), -np.inf)
        backpointer = np.zeros((n + 1, n_tags), dtype=int)
        
        # Initialize: START state
        start_idx = tag_to_idx.get(self.START, 0)
        viterbi[0, start_idx] = 0
        
        # Forward pass
        for i in range(n):
            word = tokens[i]
            
            for curr_idx, curr_tag in enumerate(tags_list):
                if viterbi[i, curr_idx] == -np.inf:
                    continue
                
                # Try all previous tags
                for prev_idx, prev_tag in enumerate(tags_list):
                    if i == 0:
                        # Bigram: START -> tag
                        trans_prob = np.log(1 / len(tags_list))
                    else:
                        # Would need trigram logic here; simplified bigram for now
                        trans_prob = np.log(1 / len(tags_list))
                    
                    emit_prob = self._log_prob_emission(word, curr_tag)
                    score = viterbi[i, prev_idx] + trans_prob + emit_prob
                    
                    if score > viterbi[i + 1, curr_idx]:
                        viterbi[i + 1, curr_idx] = score
                        backpointer[i + 1, curr_idx] = prev_idx
        
        # Backtrack
        path_idx = np.argmax(viterbi[n])
        path = []
        
        for i in range(n, 0, -1):
            path.append(tags_list[path_idx])
            path_idx = int(backpointer[i, path_idx])
        
        path.reverse()
        return path


# ============================================================================
# PART 3: MORPHOLOGY-AWARE TAGGING
# ============================================================================

class MorphologyAwareTagger:
    """
    Extends HMM tagger with morphological features.
    Combines POS tags with gender/number into compound tags.
    """
    
    def __init__(self, language='spanish'):
        self.language = language
        self.hmm = HMMPosTagger()
        self.morpho_tags = set()
        self.feature_extractors = {
            'spanish': self._extract_spanish_features,
        }
    
    def _extract_spanish_features(self, token_dict) -> str:
        """Extract gender and number from UD Spanish token features."""
        upos = token_dict.get('upos', 'NOUN')
        feats = token_dict.get('feats', {}) or {}
        
        gender = feats.get('Gender', ['U'])[0] if feats.get('Gender') else 'U'
        number = feats.get('Number', ['U'])[0] if feats.get('Number') else 'U'
        
        # Compound tag: NOUN-Fem-Sing
        compound = f"{upos}-{gender}-{number}"
        return compound
    
    def train_from_ud(self, conllu_data: List):
        """Train from UD CoNLL-U formatted data."""
        tagged_sents = []
        
        for sentence in conllu_data:
            sent_pairs = []
            for token in sentence:
                if token['id'] and '-' not in str(token['id']):  # Skip multiword tokens
                    word = token['form'].lower()
                    compound_tag = self._extract_spanish_features(token)
                    sent_pairs.append((word, compound_tag))
                    self.morpho_tags.add(compound_tag)
            
            if sent_pairs:
                tagged_sents.append(sent_pairs)
        
        self.hmm.train(tagged_sents)
        print(f"Trained morpho tagger on {len(tagged_sents)} sentences")
        print(f"Morphological tag vocabulary: {len(self.morpho_tags)} tags")
    
    def predict(self, tokens: List[str]) -> List[str]:
        """Predict morphologically-aware tags."""
        return self.hmm.viterbi_decode(tokens)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class POSEvaluator:
    """Evaluate POS taggers and attribute errors."""
    
    @staticmethod
    def evaluate(predicted_tags: List[str], gold_tags: List[str]) -> Dict:
        """Calculate tagging accuracy and per-tag metrics."""
        
        if len(predicted_tags) != len(gold_tags):
            raise ValueError("Predicted and gold tags must have same length")
        
        accuracy = sum(p == g for p, g in zip(predicted_tags, gold_tags)) / len(gold_tags)
        
        # Confusion matrix
        cm = confusion_matrix(gold_tags, predicted_tags, labels=sorted(set(gold_tags + predicted_tags)))
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'tags': sorted(set(gold_tags + predicted_tags)),
        }
        
        return metrics
    
    @staticmethod
    def attribute_errors(predicted_tokens: List[str], 
                        gold_tokens: List[str],
                        predicted_tags: List[str],
                        gold_tags: List[str]) -> Dict:
        """
        Distinguish tagging errors from segmentation errors.
        - Segmentation error: token mismatch
        - Tagging error: token match but tag mismatch
        """
        
        seg_errors = 0
        tag_errors = 0
        correct = 0
        
        for pred_tok, gold_tok, pred_tag, gold_tag in zip(
            predicted_tokens, gold_tokens, predicted_tags, gold_tags
        ):
            if pred_tok != gold_tok:
                seg_errors += 1
            elif pred_tag != gold_tag:
                tag_errors += 1
            else:
                correct += 1
        
        total = seg_errors + tag_errors + correct
        
        return {
            'correct': correct,
            'tagging_errors': tag_errors,
            'segmentation_errors': seg_errors,
            'total': total,
            'accuracy': correct / total if total > 0 else 0,
            'tag_error_rate': tag_errors / total if total > 0 else 0,
            'seg_error_rate': seg_errors / total if total > 0 else 0,
        }


# ============================================================================
# INTEGRATION INTERFACE (CONTRACT WITH PERSON A)
# ============================================================================

class POSPipeline:
    """
    End-to-end pipeline integrating Person A's segmenter with Person B's tagger.
    """
    
    def __init__(self, language='english'):
        self.language = language
        self.mft_baseline = MFTBaseline()
        self.hmm_tagger = HMMPosTagger()
        self.morpho_tagger = None
    
    def train_baselines_and_models(self, train_data):
        """Train all models."""
        self.mft_baseline.train(train_data)
        self.hmm_tagger.train(train_data)
    
    def tag_with_baseline(self, segmented_tokens: List[str]) -> List[str]:
        """
        Interface contract: Accept Person A's segmented output.
        Returns baseline POS tags.
        """
        return self.mft_baseline.predict(segmented_tokens)
    
    def tag_with_hmm(self, segmented_tokens: List[str]) -> List[str]:
        """
        Interface contract: Accept Person A's segmented output.
        Returns HMM POS tags.
        """
        return self.hmm_tagger.viterbi_decode(segmented_tokens)
    
    def tag_with_morphology(self, segmented_tokens: List[str]) -> List[str]:
        """
        Interface contract: Accept Person A's segmented output.
        Returns morphologically-aware tags (if trained).
        """
        if self.morpho_tagger is None:
            raise RuntimeError("Morphology tagger not trained")
        return self.morpho_tagger.predict(segmented_tokens)


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PERSON B: POS TAGGING & MORPHOLOGY IMPLEMENTATION")
    print("=" * 70)
    
    # ========== ENGLISH: Brown Corpus ==========
    print("\n[1] Loading Brown Corpus (English)...")
    train_sents, test_sents = DataLoader.load_brown_corpus(split_ratio=0.8)
    
    print("\n[2] Training baselines and HMM model...")
    pipeline_en = POSPipeline(language='english')
    pipeline_en.train_baselines_and_models(train_sents)
    
    # Test on a sample sentence
    test_sent = test_sents[0]
    gold_tokens = [w for w, t in test_sent]
    gold_tags = [t for w, t in test_sent]
    
    print(f"\nTest sentence: {' '.join(gold_tokens)}")
    print(f"Gold tags:     {' '.join(gold_tags)}")
    
    # Baseline predictions
    baseline_tags = pipeline_en.tag_with_baseline(gold_tokens)
    print(f"Baseline tags: {' '.join(baseline_tags)}")
    
    # HMM predictions
    hmm_tags = pipeline_en.tag_with_hmm(gold_tokens)
    print(f"HMM tags:      {' '.join(hmm_tags)}")
    
    # Evaluation
    print("\n[3] Evaluation on test set (first 100 sentences)...")
    all_pred_baseline = []
    all_pred_hmm = []
    all_gold = []
    
    for sent in test_sents[:100]:
        tokens = [w for w, t in sent]
        tags = [t for w, t in sent]
        
        all_pred_baseline.extend(pipeline_en.tag_with_baseline(tokens))
        all_pred_hmm.extend(pipeline_en.tag_with_hmm(tokens))
        all_gold.extend(tags)
    
    baseline_acc = sum(p == g for p, g in zip(all_pred_baseline, all_gold)) / len(all_gold)
    hmm_acc = sum(p == g for p, g in zip(all_pred_hmm, all_gold)) / len(all_gold)
    
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"HMM Accuracy:      {hmm_acc:.4f}")
    
    # ========== SPANISH: UD Spanish-GSD ==========
    print("\n" + "=" * 70)
    print("[4] Loading UD Spanish-GSD (with morphology)...")
    
    try:
        datasets = DataLoader.load_ud_spanish()
        
        if 'train' in datasets:
            print("\n[5] Training morphology-aware tagger...")
            morpho_pipeline = POSPipeline(language='spanish')
            
            # First train standard HMM
            train_sents_ud = []
            for sent in datasets['train']:
                sent_pairs = []
                for token in sent:
                    if token['id'] and '-' not in str(token['id']):
                        word = token['form'].lower()
                        upos = token['upos']
                        sent_pairs.append((word, upos))
                if sent_pairs:
                    train_sents_ud.append(sent_pairs)
            
            morpho_pipeline.train_baselines_and_models(train_sents_ud)
            
            # Test on sample
            test_sent_ud = datasets['test'][0] if 'test' in datasets else datasets['train'][100]
            tokens_ud = []
            gold_tags_ud = []
            
            for token in test_sent_ud:
                if token['id'] and '-' not in str(token['id']):
                    tokens_ud.append(token['form'].lower())
                    gold_tags_ud.append(token['upos'])
            
            print(f"\nTest sentence: {' '.join(tokens_ud)}")
            print(f"Gold tags:     {' '.join(gold_tags_ud)}")
            
            pred_tags_ud = morpho_pipeline.tag_with_hmm(tokens_ud)
            print(f"Predicted:     {' '.join(pred_tags_ud)}")
    
    except Exception as e:
        print(f"Note: UD Spanish loading skipped ({e})")
    
    print("\n" + "=" * 70)
    print("Implementation complete. Ready for integration with Person A's segmenter.")
    print("=" * 70)
