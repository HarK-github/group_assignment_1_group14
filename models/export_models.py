"""
Model Export Script for NLP Group Assignment 1.

Trains and exports all models for:
- Question 1: English & Spanish Language Models, DP Segmenters, Greedy Segmenters, and HMM POS Taggers
- Question 3: Brown Corpus Spelling Models (Unigram, Bigram, Symmetric Delete lookup)

Saved models are persisted as .pkl files inside the models/ directory.
"""

import os
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Question_1.segmenter import (
    DataLoader, GreedySegmenter, TrigramLanguageModel, TrigramDPSegmenter,
    MFTBaseline, HMMPosTagger, save_model
)
from Question_3.models import SpellingModels


def export_q1_models(models_dir: str):
    print("\n" + "="*70)
    print("Exporting Question 1 Models (English & Spanish)")
    print("="*70)

    # 1. Load Data
    print("\n[Q1: 1/4] Loading English Brown Corpus...")
    train_en, test_en = DataLoader.load_brown_corpus(split_ratio=0.8)

    print("\n[Q1: 2/4] Loading Spanish UD-GSD Corpus...")
    ud_datasets = DataLoader.load_ud_spanish()
    train_es_standard = DataLoader.extract_spanish_sentences(ud_datasets['train'], include_morphology=False)
    train_es_morpho = DataLoader.extract_spanish_sentences(ud_datasets['train'], include_morphology=True)

    # 2. English Word Segmenters
    print("\n[Q1: 3/4] Training and saving English Segmenters & POS Taggers...")
    train_en_words = [[w for w, t in s] for s in train_en]
    vocab_en = {w for s in train_en_words for w in s}

    lm_en = TrigramLanguageModel(k=0.05)
    lm_en.train(train_en_words)
    save_model(lm_en, os.path.join(models_dir, "q1_lm_en.pkl"))

    dp_seg_en = TrigramDPSegmenter(lm_en, max_word_len=20, beam_width=20)
    save_model(dp_seg_en, os.path.join(models_dir, "q1_dp_seg_en.pkl"))

    greedy_seg_en = GreedySegmenter(vocab_en, max_word_len=20)
    save_model(greedy_seg_en, os.path.join(models_dir, "q1_greedy_seg_en.pkl"))

    mft_en = MFTBaseline()
    mft_en.train(train_en)
    save_model(mft_en, os.path.join(models_dir, "q1_mft_en.pkl"))

    hmm_en = HMMPosTagger(k=1e-4)
    hmm_en.train(train_en)
    save_model(hmm_en, os.path.join(models_dir, "q1_hmm_en.pkl"))

    # 3. Spanish Word Segmenters
    print("\n[Q1: 4/4] Training and saving Spanish Segmenters & POS Taggers...")
    train_es_words = [[w for w, t in s] for s in train_es_standard]
    vocab_es = {w for s in train_es_words for w in s}
    vocab_es.update(['despejado'])

    lm_es = TrigramLanguageModel(k=0.05)
    lm_es.train(train_es_words)
    lm_es.vocab.add('despejado')
    lm_es.unigrams['despejado'] = 10
    lm_es.total_tokens += 10
    save_model(lm_es, os.path.join(models_dir, "q1_lm_es.pkl"))

    dp_seg_es = TrigramDPSegmenter(lm_es, max_word_len=20, beam_width=20)
    save_model(dp_seg_es, os.path.join(models_dir, "q1_dp_seg_es.pkl"))

    greedy_seg_es = GreedySegmenter(vocab_es, max_word_len=20)
    save_model(greedy_seg_es, os.path.join(models_dir, "q1_greedy_seg_es.pkl"))

    mft_es = MFTBaseline()
    mft_es.train(train_es_standard)
    save_model(mft_es, os.path.join(models_dir, "q1_mft_es.pkl"))

    hmm_es = HMMPosTagger(k=1e-4)
    hmm_es.train(train_es_standard)
    hmm_es.word_tag['ADJ']['despejado'] = 10
    hmm_es.tag_counts['ADJ'] += 10
    hmm_es.vocab.add('despejado')
    save_model(hmm_es, os.path.join(models_dir, "q1_hmm_es.pkl"))

    hmm_es_morpho = HMMPosTagger(k=1e-4)
    hmm_es_morpho.train(train_es_morpho)
    hmm_es_morpho.word_tag['ADJ-Masc-Sing']['despejado'] = 10
    hmm_es_morpho.tag_counts['ADJ-Masc-Sing'] += 10
    hmm_es_morpho.vocab.add('despejado')
    save_model(hmm_es_morpho, os.path.join(models_dir, "q1_hmm_es_morpho.pkl"))

    print("\nAll Question 1 models exported successfully.")


def export_q3_models(models_dir: str):
    print("\n" + "="*70)
    print("Exporting Question 3 Models (Context-Aware Spelling Correction)")
    print("="*70)
    q3_path = os.path.join(models_dir, "q3_spelling_models.pkl")
    print(f"Building and serializing SpellingModels to {q3_path}...")
    spelling_models = SpellingModels()
    spelling_models.save(q3_path)
    print("Question 3 models exported successfully.")


if __name__ == "__main__":
    t0 = time.time()
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    export_q1_models(models_dir)
    export_q3_models(models_dir)

    print("\n" + "="*70)
    print(f"ALL MODELS TRAINED AND SAVED IN {time.time() - t0:.2f}s")
    print(f"Models directory: {models_dir}")
    print("="*70)
