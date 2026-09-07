from models import SpellingModels

class SpellingCorrector:
    def __init__(self, models: SpellingModels):
        self.models = models

    # Part 3: Non-Word Error Correction
    def correct_non_word(self, word, use_method='B'):
        """Corrects words not found in the vocabulary."""
        if word in self.models.vocab:
            return word

        if use_method == 'A':
            candidates = self.models.method_a_candidates(word)
        else:
            candidates = self.models.method_b_candidates(word)

        if not candidates:
            return word # Fallback if no edit-distance-1 candidates exist

        # The best correction is the candidate with the highest frequency
        best_candidate = max(candidates, key=self.models.get_unigram_prob)
        return best_candidate

    # Part 3: Real-Word Error Correction
    def correct_real_word(self, phrase_tokens, target_idx, use_method='B'):
        """Checks if an in-vocabulary word is wrong in its context."""
        target_word = phrase_tokens[target_idx]
        
        if use_method == 'A':
            candidates = self.models.method_a_candidates(target_word)
        else:
            candidates = self.models.method_b_candidates(target_word)
            
        if not candidates:
            return target_word

        # Add the original word to candidates to compare against
        candidates.add(target_word)
        
        best_prob = -1
        best_candidate = target_word
        
        prev_word = phrase_tokens[target_idx - 1] if target_idx > 0 else None
        
        for candidate in candidates:
            # Calculate the probability of phrases with the corrected candidates
            if prev_word:
                prob = self.models.get_bigram_prob(prev_word, candidate)
            else:
                prob = self.models.get_unigram_prob(candidate)
                
            if prob > best_prob:
                best_prob = prob
                best_candidate = candidate

        # If a candidate phrase has a significantly higher probability, suggest it
        # (Thresholding prevents over-correction of valid but slightly less common bigrams)
        original_prob = self.models.get_bigram_prob(prev_word, target_word) if prev_word else self.models.get_unigram_prob(target_word)
        
        if best_prob > (original_prob * 1.5): # 1.5x threshold for "significantly higher"
            return best_candidate
            
        return target_word