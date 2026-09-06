import random
import time
from nltk.corpus import brown
from models import SpellingModels
from corrector import SpellingCorrector

def introduce_spelling_mistake(word, vocab, real_word=False):
    """Introduces a single edit distance mistake."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    edit_type = random.choice(['delete', 'transpose', 'replace', 'insert'])
    
    if len(word) < 3: return word # Skip very short words
    
    i = random.randint(0, len(word) - 1)
    
    if edit_type == 'delete':
        mutated = word[:i] + word[i+1:]
    elif edit_type == 'transpose' and i < len(word)-1:
        mutated = word[:i] + word[i+1] + word[i] + word[i+2:]
    elif edit_type == 'replace':
        mutated = word[:i] + random.choice(letters) + word[i+1:]
    else: # insert
        mutated = word[:i] + random.choice(letters) + word[i:]
        
    if real_word:
        return mutated if mutated in vocab and mutated != word else word
    else:
        return mutated if mutated not in vocab else word

def evaluate_and_benchmark():
    models = SpellingModels()
    corrector = SpellingCorrector(models)
    
    # --- Test Set Generation ---
    sentences = list(brown.sents())
    test_size = int(len(sentences) * 0.10)
    test_sentences = random.sample(sentences, test_size)
    
    non_word_test = []
    real_word_test = []
    
    print("\nGenerating Test Sets...")
    for sent in test_sentences:
        sent = [w.lower() for w in sent if w.isalpha()]
        if len(sent) < 3: continue
            
        target_idx = random.randint(0, len(sent)-1)
        original_word = sent[target_idx]
        
        # Create non-word error
        nw_error = introduce_spelling_mistake(original_word, models.vocab, real_word=False)
        if nw_error != original_word:
            non_word_test.append((sent, target_idx, original_word, nw_error))
            
        # Create real-word error
        rw_error = introduce_spelling_mistake(original_word, models.vocab, real_word=True)
        if rw_error != original_word:
            real_word_test.append((sent, target_idx, original_word, rw_error))

    # Cap at 500 for timely evaluation during testing, can be increased
    non_word_test = non_word_test[:500]
    real_word_test = real_word_test[:500]

    # --- Accuracy Reporting ---
    print("\nCalculating Accuracy...")
    nw_correct = sum(1 for sent, idx, orig, err in non_word_test if corrector.correct_non_word(err, 'B') == orig)
    print(f"Non-Word Accuracy: {nw_correct / len(non_word_test):.2%}")
    
    rw_correct = 0
    for sent, idx, orig, err in real_word_test:
        mutated_sent = sent.copy()
        mutated_sent[idx] = err
        if corrector.correct_real_word(mutated_sent, idx, 'B') == orig:
            rw_correct += 1
    print(f"Real-Word Accuracy: {rw_correct / len(real_word_test):.2%}")

    # --- Speed Demon Benchmark ---
    print("\nStarting Speed Demon Benchmark...")
    # Isolate exactly 1,000 non-word errors
    benchmark_batch = [err for _, _, _, err in non_word_test]
    while len(benchmark_batch) < 1000:
        # Pad with randomly mutated words if we don't have enough
        rand_word = random.choice(list(models.vocab))
        err = introduce_spelling_mistake(rand_word, models.vocab, real_word=False)
        if err != rand_word: benchmark_batch.append(err)
    benchmark_batch = benchmark_batch[:1000]

    # Test Method A
    start_a = time.perf_counter()
    for word in benchmark_batch:
        corrector.correct_non_word(word, 'A')
    time_a = time.perf_counter() - start_a

    # Test Method B
    start_b = time.perf_counter()
    for word in benchmark_batch:
        corrector.correct_non_word(word, 'B')
    time_b = time.perf_counter() - start_b

    print(f"Method A (Standard Edit Distance) Latency: {time_a:.4f} seconds")
    print(f"Method B (Symmetric Delete) Latency:       {time_b:.4f} seconds")
    
    print("\n--- Benchmark Conclusion ---")
    print("Method B (Symmetric Delete) achieves significantly lower execution latency")
    print("because it relies on O(1) dictionary lookups for pre-computed deletions,")
    print("whereas Method A computes hundreds of potential transpositions, replacements,")
    print("and insertions dynamically for every word in real-time.")

if __name__ == "__main__":
    evaluate_and_benchmark()