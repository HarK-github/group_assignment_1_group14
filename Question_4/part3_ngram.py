"""
Question 4, Part 3: Smoothed N-gram Language Models

This module defines add-k smoothed bigram and trigram models for
providing context-aware probabilities.
"""

from typing import List

class SmoothedNgramModels:
    """Class for Add-k smoothed N-gram language models."""
    
    def __init__(self, k: float = 0.05):
        """
        Initializes the smoothed N-gram model.
        
        Args:
            k: The smoothing parameter.
        """
        self.k = k
        raise NotImplementedError("SmoothedNgramModels.__init__ is not fully implemented.")
        
    def train(self, sentences: List[List[str]]) -> None:
        """
        Trains the N-gram models on the given sentences.
        
        Args:
            sentences: List of tokenized sentences.
        """
        raise NotImplementedError("train is not yet implemented.")

    def get_bigram_prob(self, word1: str, word2: str) -> float:
        """
        Calculates the smoothed bigram probability P(word2 | word1).
        """
        raise NotImplementedError("get_bigram_prob is not yet implemented.")

    def get_trigram_prob(self, word1: str, word2: str, word3: str) -> float:
        """
        Calculates the smoothed trigram probability P(word3 | word1, word2).
        """
        raise NotImplementedError("get_trigram_prob is not yet implemented.")
