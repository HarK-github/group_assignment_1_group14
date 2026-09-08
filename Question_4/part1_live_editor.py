import os
import sys
import random
import time
import pickle
import nltk
from nltk.corpus import brown

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown', quiet=True)

# Add Question_1 and Question_3 to path so we can import their modules for pickle loading
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'Question_1'))
sys.path.append(os.path.join(parent_dir, 'Question_3'))
from Question_1.segmenter import TrigramDPSegmenter, TrigramLanguageModel
from Question_3.models import SpellingModels
from Question_3.corrector import SpellingCorrector

class PassageSampler:
    """Samples paragraphs from a corpus for simulated typing."""
    
    @staticmethod
    def sample_paragraph(min_sentences=5, max_sentences=8) -> str:
        """
        Samples a paragraph of 5-8 contiguous sentences from the Brown corpus.
        Returns the paragraph as a single string.
        """
        sents = brown.sents()
        num_sents = random.randint(min_sentences, max_sentences)
        
        # Pick a random starting point
        max_start = len(sents) - num_sents
        start_idx = random.randint(0, max_start)
        
        sampled = sents[start_idx : start_idx + num_sents]
        
        # Detokenize slightly for natural text (handling punctuation spacing is complex, but basic join is fine for now)
        paragraph = ""
        for sent in sampled:
            sent_str = " ".join(sent)
            # Basic cleanup for punctuation
            sent_str = sent_str.replace(" .", ".").replace(" ,", ",").replace(" '", "'").replace(" ?", "?").replace(" !", "!")
            paragraph += sent_str + " "
            
        return paragraph.strip()

class MergeGenerator:
    """Simulates fast-typing errors like space omissions."""
    
    @staticmethod
    def simulate_typing(text: str, merge_prob: float = 0.08) -> list:
        """
        Takes a string and yields it token by token, dropping spaces
        between consecutive words with probability p.
        
        p = 0.08 justifies modeling common keyboard typing errors where 
        a space bar stroke is missed during fast, continuous typing.
        """
        words = text.split()
        if not words:
            return []
            
        simulated_tokens = []
        current_token = words[0]
        
        for i in range(1, len(words)):
            # Drop space with probability merge_prob
            if random.random() < merge_prob:
                # Merge the next word into the current token
                current_token += words[i]
            else:
                simulated_tokens.append(current_token)
                current_token = words[i]
                
        simulated_tokens.append(current_token)
        return simulated_tokens

class EditorPipeline:
    """Core logic for live-typing simulation and alert generation."""
    
    def __init__(self, models_dir: str):
        self.trigger_interval = 5 # N=5 balances real-time feedback cost with context window size for trigram/bigram models
        
        print("Loading models for the Editor Pipeline...")
        # Graceful loading of models
        self.segmenter = None
        self.spelling_models = None
        self.corrector = None
        self.lm = None
        
        # Load Question 1 DP Segmenter
        q1_seg_path = os.path.join(models_dir, 'q1_dp_seg_en.pkl')
        if os.path.exists(q1_seg_path):
            with open(q1_seg_path, 'rb') as f:
                self.segmenter = pickle.load(f)
                self.lm = self.segmenter.lm
                # Fallback to initialize lm manually if pickling structure differs
                if not hasattr(self.segmenter, 'lm'):
                     q1_lm_path = os.path.join(models_dir, 'q1_lm_en.pkl')
                     if os.path.exists(q1_lm_path):
                         with open(q1_lm_path, 'rb') as lf:
                             self.segmenter.lm = pickle.load(lf)
                             self.lm = self.segmenter.lm
        else:
            print(f"Warning: Segmenter model not found at {q1_seg_path}. Pipeline will use mock behavior.")
            
        # Load Question 3 Spelling Models
        q3_spell_path = os.path.join(models_dir, 'q3_spelling_models.pkl')
        if os.path.exists(q3_spell_path):
            with open(q3_spell_path, 'rb') as f:
                self.spelling_models = pickle.load(f)
                self.corrector = SpellingCorrector(self.spelling_models)
        else:
            print(f"Warning: Spelling model not found at {q3_spell_path}. Pipeline will use mock behavior.")

        self.token_times = []
        self.trigger_times = []
        self.processed_words = []

    def get_latencies(self):
        """Returns average latency in milliseconds."""
        avg_token = (sum(self.token_times) / len(self.token_times) * 1000) if self.token_times else 0
        avg_trigger = (sum(self.trigger_times) / len(self.trigger_times) * 1000) if self.trigger_times else 0
        return round(avg_token, 2), round(avg_trigger, 2)

    def process_token(self, token: str) -> dict:
        """Processes a single token as it is 'typed'."""
        start_time = time.time()
        clean_token = ''.join(e for e in token if e.isalnum()).lower()
        alert = None
        corrected_token = token
        is_oov = False
        
        if not clean_token:
             # Punctuation or empty string, just append and return
             self.processed_words.append(token)
             return {'token': token, 'alert': None}

        # 1. [SEGMENT-ALERT] Check
        if self.spelling_models and self.segmenter:
            if clean_token not in self.spelling_models.vocab or len(clean_token) > 15:
                is_oov = True
                segmented = self.segmenter.segment(clean_token)
                # If segmenter split the token into valid words
                if len(segmented) > 1 and all(w in self.spelling_models.vocab for w in segmented):
                    corrected_token = " ".join(segmented)
                    # Attempt to preserve original capitalization if possible (basic)
                    if token.istitle():
                        corrected_token = corrected_token.capitalize()
                    
                    # Also append any trailing punctuation
                    trailing_punct = token[len(clean_token):]
                    if trailing_punct and not trailing_punct.isalnum():
                        corrected_token += trailing_punct
                        
                    alert = f"[SEGMENT-ALERT] Merged words detected. Suggested: {corrected_token}"
                    is_oov = False # Successfully segmented

        # 2. [SPELL-ALERT] Check (if still OOV and not segmented)
        if self.corrector and is_oov and not alert:
            correction = self.corrector.correct_non_word(clean_token, use_method='B')
            if correction != clean_token:
                # Maintain case and punctuation if possible
                corrected_token = correction.capitalize() if token.istitle() else correction
                trailing_punct = token[len(clean_token):]
                if trailing_punct and not trailing_punct.isalnum():
                    corrected_token += trailing_punct
                alert = f"[SPELL-ALERT] Misspelled word. Suggested: {corrected_token}"

        token_duration = time.time() - start_time
        self.token_times.append(token_duration)
        
        # Use the corrected token for context going forward
        self.processed_words.append(corrected_token)
        
        # 3. Trigger-Based Checks (Every N words)
        if len(self.processed_words) % self.trigger_interval == 0:
            start_trigger_time = time.time()
            trigger_alert = self._run_trigger_checks()
            trigger_duration = time.time() - start_trigger_time
            self.trigger_times.append(trigger_duration)
            
            # Combine alerts if needed (prioritizing the token alert)
            if trigger_alert:
                if alert:
                    alert = alert + " | " + trigger_alert
                else:
                    alert = trigger_alert

        return {'token': corrected_token, 'alert': alert}
        
    def _run_trigger_checks(self) -> str:
        """Runs grammar and real-word error checks on the recent context window."""
        window = self.processed_words[-self.trigger_interval:]
        clean_window = [''.join(e for e in w if e.isalnum()).lower() for w in window]
        clean_window = [w for w in clean_window if w] # Remove empty strings
        
        if len(clean_window) < 3:
            return None
            
        alerts = []
        
        # [GRAMMAR-ALERT]: Trigram perplexity / log probability
        if self.lm:
            # We check the log prob of the last word given the two previous words
            w3 = clean_window[-1]
            w2 = clean_window[-2]
            w1 = clean_window[-3]
            log_prob = self.lm.log_prob(w3, w1, w2) # Note: log_prob(w, w_prev2, w_prev1) from segmenter.py
            
            # Very low log_prob implies implausible grammar
            # Threshold chosen empirically (e.g. -15.0 or lower)
            if log_prob < -15.0:
                 alerts.append(f"[GRAMMAR-ALERT] Implausible sequence: '{w1} {w2} {w3}'")
                 
        # Real-Word Error Alert
        if self.corrector and len(clean_window) >= 2:
            target_idx = len(clean_window) - 1
            correction = self.corrector.correct_real_word(clean_window, target_idx, use_method='B')
            if correction != clean_window[target_idx]:
                alerts.append(f"[GRAMMAR-ALERT] Contextual error. '{clean_window[target_idx]}' -> suggested: {correction}")
                
        if alerts:
            return " | ".join(alerts)
        return None
