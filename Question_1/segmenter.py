import os
import sys
import re
import math
import time
import pickle
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np
import requests
import nltk
from nltk.corpus import brown
import conllu
from sklearn.metrics import confusion_matrix


# =====================================================================
# 1. Data Loader
# =====================================================================

class DataLoader:
    """Data loader and preprocessor for English (Brown) and Spanish (UD-GSD)."""
    
    @staticmethod
    def load_brown_corpus(split_ratio: float = 0.8) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
        """Loads and cleans the English Brown corpus tagged with Universal tagset."""
        nltk.download('brown', quiet=True)
        nltk.download('universal_tagset', quiet=True)
        
        raw_sents = brown.tagged_sents(tagset='universal')
        cleaned_sents = []
        for sent in raw_sents:
            clean_pairs = []
            for word, tag in sent:
                clean_word = re.sub(r'[^a-zA-Z0-9]', '', word).lower()
                if clean_word:
                    clean_pairs.append((clean_word, tag))
            if clean_pairs:
                cleaned_sents.append(clean_pairs)
                
        split_idx = int(len(cleaned_sents) * split_ratio)
        train_sents = cleaned_sents[:split_idx]
        test_sents = cleaned_sents[split_idx:]
        print(f"Brown Corpus (English): {len(train_sents):,} train, {len(test_sents):,} test sentences loaded.")
        return train_sents, test_sents

    @staticmethod
    def load_ud_spanish(cache_dir: Optional[str] = None) -> Dict[str, Any]:
        """Loads the Universal Dependencies Spanish-GSD corpus with local caching."""
        url = "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/master"
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ud_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        datasets = {}
        for split in ['train', 'dev', 'test']:
            cache_file = os.path.join(cache_dir, f"es_gsd-ud-{split}.conllu")
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                file_url = f"{url}/es_gsd-ud-{split}.conllu"
                resp = requests.get(file_url, timeout=45)
                resp.raise_for_status()
                content = resp.text
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(content)
            datasets[split] = conllu.parse(content)
            print(f"UD Spanish-GSD ({split}): {len(datasets[split]):,} sentences loaded.")
        return datasets

    @staticmethod
    def extract_spanish_sentences(conllu_sentences, include_morphology: bool = False) -> List[List[Tuple[str, str]]]:
        """Extracts cleaned (word, tag) pairs from CoNLL-U parsed structures."""
        extracted = []
        for sent in conllu_sentences:
            pairs = []
            for token in sent:
                # Skip multiword tokens and standalone punctuation
                if token['id'] and isinstance(token['id'], int) and token['upos'] != 'PUNCT':
                    raw_word = token['form'].lower()
                    # Retain alphanumeric characters and Spanish diacritics
                    word = re.sub(r'[^a-záéíóúüñ0-9]', '', raw_word)
                    if not word:
                        continue
                    if include_morphology:
                        feats = token.get('feats') or {}
                        gender = feats.get('Gender', 'None')
                        number = feats.get('Number', 'None')
                        tag = f"{token['upos']}-{gender}-{number}"
                    else:
                        tag = token['upos']
                    pairs.append((word, tag))
            if pairs:
                extracted.append(pairs)
        return extracted


# =====================================================================
# 2. Word Segmentation
# =====================================================================

class GreedySegmenter:
    """Greedy longest-match baseline segmenter."""
    def __init__(self, vocab: Set[str], max_word_len: int = 20):
        self.vocab = set(vocab)
        self.max_word_len = max_word_len

    def segment(self, s: str) -> List[str]:
        if not s:
            return []
        tokens = []
        i = 0
        n = len(s)
        while i < n:
            matched = False
            for l in range(min(self.max_word_len, n - i), 0, -1):
                sub = s[i : i + l]
                if sub in self.vocab:
                    tokens.append(sub)
                    i += l
                    matched = True
                    break
            if not matched:
                tokens.append(s[i])
                i += 1
        return tokens


class TrigramLanguageModel:
    """Trigram Language Model with Laplace smoothing and backoff."""
    def __init__(self, k: float = 0.05):
        self.k = k
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocab = set()
        self.total_tokens = 0
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'

    def train(self, sentences: List[List[str]]):
        """Trains unigram, bigram, and trigram counts over tokenized sentences."""
        for sent in sentences:
            padded = [self.BOS, self.BOS] + sent + [self.EOS]
            for w in sent:
                self.unigrams[w] += 1
                self.vocab.add(w)
            self.unigrams[self.BOS] += 2
            self.unigrams[self.EOS] += 1
            self.total_tokens += len(sent)

            for i in range(len(padded) - 1):
                self.bigrams[(padded[i], padded[i+1])] += 1

            for i in range(len(padded) - 2):
                self.trigrams[(padded[i], padded[i+1], padded[i+2])] += 1

    def log_prob(self, w: str, w_prev2: str, w_prev1: str) -> float:
        """Computes log probability with Laplace smoothing and backoff."""
        V = len(self.vocab) + 1
        tri_count = self.trigrams.get((w_prev2, w_prev1, w), 0)
        bi_count = self.bigrams.get((w_prev2, w_prev1), 0)
        
        # Trigram probability
        if bi_count > 0:
            p_tri = (tri_count + self.k) / (bi_count + self.k * V)
            return math.log(p_tri)
        
        # Backoff to Bigram
        bi_ctx_count = self.unigrams.get(w_prev1, 0)
        bi_pair_count = self.bigrams.get((w_prev1, w), 0)
        if bi_ctx_count > 0:
            p_bi = (bi_pair_count + self.k) / (bi_ctx_count + self.k * V)
            return math.log(p_bi) - 1.0
        
        # Backoff to Unigram
        uni_count = self.unigrams.get(w, 0)
        p_uni = (uni_count + self.k) / (self.total_tokens + self.k * V)
        return math.log(p_uni) - 2.0


class TrigramDPSegmenter:
    """Viterbi Dynamic Programming Word Segmenter with Beam Search."""
    def __init__(self, lm: TrigramLanguageModel, max_word_len: int = 20, beam_width: int = 20):
        self.lm = lm
        self.vocab = lm.vocab
        self.max_word_len = max_word_len
        self.beam_width = beam_width

    def segment(self, s: str) -> List[str]:
        """Segments an unspaced string into words via Viterbi DP + Beam Search."""
        if not s:
            return []
        n = len(s)
        # dp[j]: list of tuples (score, w_prev1, w_curr, prev_pos, prev_hyp_idx)
        dp = defaultdict(list)
        dp[0] = [(0.0, self.lm.BOS, self.lm.BOS, -1, -1)]

        for j in range(1, n + 1):
            candidates = []
            min_i = max(0, j - self.max_word_len)
            for i in range(min_i, j):
                if not dp[i]:
                    continue
                w = s[i:j]
                in_vocab = w in self.vocab
                
                # Single character OOV fallback penalty
                if not in_vocab:
                    if len(w) == 1:
                        oov_pen = -18.0
                    else:
                        continue # Multi-character candidates must be in vocabulary
                else:
                    oov_pen = 0.0

                for hyp_idx, (score, w_prev2, w_prev1, _, _) in enumerate(dp[i]):
                    if in_vocab:
                        lp = self.lm.log_prob(w, w_prev2, w_prev1)
                    else:
                        lp = oov_pen
                    total_score = score + lp
                    candidates.append((total_score, w_prev1, w, i, hyp_idx))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                seen_pairs = set()
                pruned = []
                for cand in candidates:
                    pair = (cand[1], cand[2])
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        pruned.append(cand)
                    if len(pruned) >= self.beam_width:
                        break
                dp[j] = pruned

        if not dp[n]:
            return list(s)

        # Backtrack optimal hypothesis
        best_cand = dp[n][0]
        tokens = []
        curr_pos = n
        curr_hyp = best_cand

        while curr_pos > 0:
            tokens.append(curr_hyp[2])
            prev_pos = curr_hyp[3]
            prev_hyp_idx = curr_hyp[4]
            if prev_pos <= 0:
                break
            curr_hyp = dp[prev_pos][prev_hyp_idx]
            curr_pos = prev_pos

        tokens.reverse()
        return tokens


class SegmentationMetrics:
    """Calculates boundary cut metrics and exact sentence match rate."""
    @staticmethod
    def get_boundaries(tokens: List[str]) -> Set[int]:
        boundaries = set()
        pos = 0
        for t in tokens[:-1]:
            pos += len(t)
            boundaries.add(pos)
        return boundaries

    @classmethod
    def evaluate_boundaries(cls, pred_tokens: List[str], gold_tokens: List[str]) -> Tuple[float, float, float, bool]:
        pred_b = cls.get_boundaries(pred_tokens)
        gold_b = cls.get_boundaries(gold_tokens)
        is_exact = (pred_tokens == gold_tokens)
        
        if not pred_b and not gold_b:
            return 1.0, 1.0, 1.0, is_exact
        if not pred_b or not gold_b:
            return 0.0, 0.0, 0.0, is_exact
            
        tp = len(pred_b & gold_b)
        precision = tp / len(pred_b) if pred_b else 0.0
        recall = tp / len(gold_b) if gold_b else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1, is_exact


# =====================================================================
# 3. POS Tagging & Morphology Extension
# =====================================================================

class MFTBaseline:
    """Most-Frequent-Tag baseline POS tagger."""
    def __init__(self, fallback_tag: str = 'NOUN'):
        self.word_tags = defaultdict(Counter)
        self.tag_counts = Counter()
        self.fallback_tag = fallback_tag

    def train(self, tagged_sentences: List[List[Tuple[str, str]]]):
        for sent in tagged_sentences:
            for word, tag in sent:
                w_lower = word.lower()
                self.word_tags[w_lower][tag] += 1
                self.tag_counts[tag] += 1
        if self.tag_counts:
            self.fallback_tag = self.tag_counts.most_common(1)[0][0]

    def predict(self, tokens: List[str]) -> List[str]:
        tags = []
        for tok in tokens:
            w_lower = tok.lower()
            if w_lower in self.word_tags:
                tags.append(self.word_tags[w_lower].most_common(1)[0][0])
            else:
                tags.append(self.fallback_tag)
        return tags


class HMMPosTagger:
    """First-order Hidden Markov Model POS tagger with Viterbi decoding."""
    def __init__(self, k: float = 1e-4):
        self.k = k
        self.tag_unigrams = Counter()
        self.tag_bigrams = Counter()
        self.word_tag = defaultdict(Counter)
        self.tag_counts = Counter()
        self.tags = set()
        self.vocab = set()
        self.START = '<START>'
        self.END = '<END>'

    def train(self, tagged_sentences: List[List[Tuple[str, str]]]):
        """Trains transition and emission distributions from tagged sentences."""
        for sent in tagged_sentences:
            words = [w.lower() for w, t in sent]
            tags = [t for w, t in sent]
            padded_tags = [self.START] + tags + [self.END]

            for i in range(len(padded_tags) - 1):
                t_prev, t_curr = padded_tags[i], padded_tags[i+1]
                self.tag_unigrams[t_curr] += 1
                self.tag_bigrams[(t_prev, t_curr)] += 1
                
                if i < len(words):
                    w = words[i]
                    self.word_tag[t_curr][w] += 1
                    self.tag_counts[t_curr] += 1
                    self.vocab.add(w)
            self.tags.update(tags)

    def _log_prob_transition(self, t_prev: str, t_curr: str) -> float:
        V_tags = len(self.tags)
        pair_count = self.tag_bigrams.get((t_prev, t_curr), 0)
        prev_count = self.tag_unigrams.get(t_prev, 0)
        return math.log((pair_count + self.k) / (prev_count + self.k * V_tags))

    def _log_prob_emission(self, word: str, tag: str) -> float:
        V_words = len(self.vocab)
        word_count = self.word_tag[tag].get(word, 0)
        tag_total = self.tag_counts[tag]
        return math.log((word_count + self.k) / (tag_total + self.k * (V_words + 1)))

    def viterbi_decode(self, tokens: List[str]) -> List[str]:
        """Finds most probable tag sequence via Viterbi trellis dynamic programming."""
        if not tokens:
            return []
        tokens = [t.lower() for t in tokens]
        n = len(tokens)
        tags_list = sorted(list(self.tags))
        n_tags = len(tags_list)

        viterbi = np.full((n, n_tags), -np.inf)
        backpointer = np.zeros((n, n_tags), dtype=int)

        # Step 0: START -> Tag
        for tag_idx, tag in enumerate(tags_list):
            trans = self._log_prob_transition(self.START, tag)
            emit = self._log_prob_emission(tokens[0], tag)
            viterbi[0, tag_idx] = trans + emit

        # Trellis induction
        for i in range(1, n):
            word = tokens[i]
            for curr_idx, curr_tag in enumerate(tags_list):
                emit = self._log_prob_emission(word, curr_tag)
                best_score = -np.inf
                best_prev = 0
                for prev_idx, prev_tag in enumerate(tags_list):
                    trans = self._log_prob_transition(prev_tag, curr_tag)
                    score = viterbi[i - 1, prev_idx] + trans + emit
                    if score > best_score:
                        best_score = score
                        best_prev = prev_idx
                viterbi[i, curr_idx] = best_score
                backpointer[i, curr_idx] = best_prev

        # Backtrack
        best_last_idx = int(np.argmax(viterbi[n - 1]))
        path = [best_last_idx]
        for i in range(n - 1, 0, -1):
            path.append(backpointer[i, path[-1]])
        path.reverse()
        return [tags_list[idx] for idx in path]


# =====================================================================
# 4. Error Attribution & Evaluation Helpers
# =====================================================================

class AlignedErrorAttributor:
    """Attributes errors to either word boundary segmentation or genuine POS tagging."""
    @staticmethod
    def get_token_spans(tokens: List[str]) -> List[Tuple[int, int, str]]:
        spans = []
        pos = 0
        for t in tokens:
            spans.append((pos, pos + len(t), t))
            pos += len(t)
        return spans

    @classmethod
    def attribute_errors(cls, pred_tokens: List[str], gold_tokens: List[str],
                         pred_tags: List[str], gold_tags: List[str]) -> Dict[str, Any]:
        pred_spans = cls.get_token_spans(pred_tokens)
        gold_spans = cls.get_token_spans(gold_tokens)

        gold_map = { (start, end): (tok, tag) for (start, end, tok), tag in zip(gold_spans, gold_tags) }
        pred_map = { (start, end): (tok, tag) for (start, end, tok), tag in zip(pred_spans, pred_tags) }

        correct = 0
        tagging_errors = 0
        seg_errors = 0

        for (g_start, g_end), (g_tok, g_tag) in gold_map.items():
            if (g_start, g_end) in pred_map:
                p_tok, p_tag = pred_map[(g_start, g_end)]
                if p_tag == g_tag:
                    correct += 1
                else:
                    tagging_errors += 1
            else:
                seg_errors += 1

        total_gold = len(gold_tokens)
        return {
            'total_gold_tokens': total_gold,
            'correct': correct,
            'genuine_tagging_errors': tagging_errors,
            'segmentation_caused_errors': seg_errors,
            'end_to_end_accuracy': correct / total_gold if total_gold else 0.0,
            'genuine_tag_error_rate': tagging_errors / total_gold if total_gold else 0.0,
            'seg_caused_error_rate': seg_errors / total_gold if total_gold else 0.0
        }


def run_benchmark(lang_name: str, eval_data: List[List[Tuple[str, str]]],
                  greedy_seg: GreedySegmenter, dp_seg: TrigramDPSegmenter,
                  mft_tagger: MFTBaseline, hmm_tagger: HMMPosTagger) -> Dict[str, Any]:
    """Runs a complete benchmark over evaluation sentences measuring segmentation and tagging."""
    print(f"Evaluating {lang_name} across {len(eval_data)} unspaced test sentences...")
    greedy_p_list, greedy_r_list, greedy_f1_list, greedy_ex_list = [], [], [], []
    dp_p_list, dp_r_list, dp_f1_list, dp_ex_list = [], [], [], []
    
    gold_tok_acc_mft = []
    gold_tok_acc_hmm = []
    
    greedy_attr_list = []
    dp_attr_list = []
    
    all_gold_tags_flat = []
    all_hmm_tags_flat = []

    t_start = time.time()
    for sent in eval_data:
        gold_tokens = [w for w, t in sent]
        gold_tags = [t for w, t in sent]
        unspaced_str = "".join(gold_tokens)
        
        # 1. Segmentation
        greedy_tokens = greedy_seg.segment(unspaced_str)
        dp_tokens = dp_seg.segment(unspaced_str)
        
        gp, gr, gf1, gex = SegmentationMetrics.evaluate_boundaries(greedy_tokens, gold_tokens)
        greedy_p_list.append(gp); greedy_r_list.append(gr); greedy_f1_list.append(gf1); greedy_ex_list.append(gex)
        
        dp_p, dp_r, dp_f1, dp_ex = SegmentationMetrics.evaluate_boundaries(dp_tokens, gold_tokens)
        dp_p_list.append(dp_p); dp_r_list.append(dp_r); dp_f1_list.append(dp_f1); dp_ex_list.append(dp_ex)
        
        # 2. POS Tagging on Gold Tokens
        mft_pred_gold = mft_tagger.predict(gold_tokens)
        hmm_pred_gold = hmm_tagger.viterbi_decode(gold_tokens)
        gold_tok_acc_mft.append(sum(p == g for p, g in zip(mft_pred_gold, gold_tags)) / len(gold_tags))
        gold_tok_acc_hmm.append(sum(p == g for p, g in zip(hmm_pred_gold, gold_tags)) / len(gold_tags))
        
        all_gold_tags_flat.extend(gold_tags)
        all_hmm_tags_flat.extend(hmm_pred_gold)
        
        # 3. End-to-End Pipeline & Error Attribution
        greedy_pred_tags = hmm_tagger.viterbi_decode(greedy_tokens)
        dp_pred_tags = hmm_tagger.viterbi_decode(dp_tokens)
        
        greedy_attr = AlignedErrorAttributor.attribute_errors(greedy_tokens, gold_tokens, greedy_pred_tags, gold_tags)
        dp_attr = AlignedErrorAttributor.attribute_errors(dp_tokens, gold_tokens, dp_pred_tags, gold_tags)
        
        greedy_attr_list.append(greedy_attr)
        dp_attr_list.append(dp_attr)

    elapsed = time.time() - t_start
    print(f"  Benchmark completed in {elapsed:.2f}s")
    
    return {
        'greedy_seg': {
            'precision': float(np.mean(greedy_p_list)),
            'recall': float(np.mean(greedy_r_list)),
            'f1': float(np.mean(greedy_f1_list)),
            'exact_match': float(np.mean(greedy_ex_list))
        },
        'dp_seg': {
            'precision': float(np.mean(dp_p_list)),
            'recall': float(np.mean(dp_r_list)),
            'f1': float(np.mean(dp_f1_list)),
            'exact_match': float(np.mean(dp_ex_list))
        },
        'gold_tagging': {
            'mft_acc': float(np.mean(gold_tok_acc_mft)),
            'hmm_acc': float(np.mean(gold_tok_acc_hmm))
        },
        'greedy_e2e': {
            'accuracy': float(np.mean([a['end_to_end_accuracy'] for a in greedy_attr_list])),
            'genuine_tag_err': float(np.mean([a['genuine_tag_error_rate'] for a in greedy_attr_list])),
            'seg_caused_err': float(np.mean([a['seg_caused_error_rate'] for a in greedy_attr_list]))
        },
        'dp_e2e': {
            'accuracy': float(np.mean([a['end_to_end_accuracy'] for a in dp_attr_list])),
            'genuine_tag_err': float(np.mean([a['genuine_tag_error_rate'] for a in dp_attr_list])),
            'seg_caused_err': float(np.mean([a['seg_caused_error_rate'] for a in dp_attr_list]))
        },
        'confusion_matrix': (all_gold_tags_flat, all_hmm_tags_flat)
    }


def print_confusion_matrix_ascii(gold_tags: List[str], pred_tags: List[str], title: str = "Confusion Matrix"):
    """Prints a clean ASCII confusion matrix for the most common POS tags."""
    tag_counts = Counter(gold_tags)
    top_labels = [tag for tag, count in tag_counts.most_common(8)]
    cm = confusion_matrix(gold_tags, pred_tags, labels=top_labels)
    
    print(f"\n{'='*70}")
    print(f"{title} (Top Tags: Actual rows vs. Predicted columns)")
    print(f"{'='*70}")
    
    header = f"{'Actual \\ Pred':<15}" + "".join([f"{l[:6]:>8}" for l in top_labels])
    print(header)
    print("-" * len(header))
    for i, label in enumerate(top_labels):
        row = f"{label:<15}" + "".join([f"{cm[i, j]:>8}" for j in range(len(top_labels))])
        print(row)


# =====================================================================
# 5. Model Serialization & Deserialization
# =====================================================================

def save_model(model: Any, filepath: str):
    """Saves a model object to a pickle file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model successfully saved to: {filepath}")


def load_model(filepath: str) -> Any:
    """Loads a model object from a pickle file."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model successfully loaded from: {filepath}")
    return model
