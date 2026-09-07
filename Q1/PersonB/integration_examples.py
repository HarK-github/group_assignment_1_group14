"""
INTEGRATION EXAMPLES: Person A's Segmenter → Person B's POS Tagger
Demonstrates the interface contract and end-to-end pipeline.
"""

from pos_morphology_tagger import (
    DataLoader, MFTBaseline, HMMPosTagger, MorphologyAwareTagger,
    POSEvaluator, POSPipeline
)


# ============================================================================
# EXAMPLE 1: MOCK DATA TESTING (Before Person A's Segmenter is Ready)
# ============================================================================

def example_1_mock_data():
    """
    Person B can start immediately using ground-truth tokens from corpora
    while Person A finishes their segmentation implementation.
    """
    print("=" * 70)
    print("EXAMPLE 1: Mock Data Testing (Gold Standard Tokens)")
    print("=" * 70)
    
    # Load training data
    train_sents, test_sents = DataLoader.load_brown_corpus()
    
    # Train models
    pipeline = POSPipeline(language='english')
    pipeline.train_baselines_and_models(train_sents)
    
    # Use gold-standard tokens from test set
    test_sent = test_sents[5]
    gold_tokens = [w for w, t in test_sent]
    gold_tags = [t for w, t in test_sent]
    
    print(f"\nGold tokens:  {gold_tokens}")
    print(f"Gold tags:    {gold_tags}")
    
    # Test both methods
    baseline_pred = pipeline.tag_with_baseline(gold_tokens)
    hmm_pred = pipeline.tag_with_hmm(gold_tokens)
    
    print(f"\nMFT Baseline: {baseline_pred}")
    print(f"HMM Model:    {hmm_pred}")
    
    # Compare accuracies
    baseline_correct = sum(p == g for p, g in zip(baseline_pred, gold_tags))
    hmm_correct = sum(p == g for p, g in zip(hmm_pred, gold_tags))
    
    print(f"\nBaseline accuracy: {baseline_correct}/{len(gold_tags)}")
    print(f"HMM accuracy:      {hmm_correct}/{len(gold_tags)}")


# ============================================================================
# EXAMPLE 2: INTERFACE CONTRACT - ACCEPTING PERSON A'S OUTPUT
# ============================================================================

def example_2_integration_day():
    """
    On 'integration day', connect Person A's segmenter output directly.
    This demonstrates the defined interface contract.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Integration with Person A's Segmenter")
    print("=" * 70)
    
    # Train models (unchanged)
    train_sents, _ = DataLoader.load_brown_corpus()
    pipeline = POSPipeline(language='english')
    pipeline.train_baselines_and_models(train_sents)
    
    # --- INTERFACE CONTRACT: Person A returns this format ---
    # Simulating Person A's segmenter output: list[str]
    person_a_output = [
        "the", "quick", "brown", "fox",
        "jumps", "over", "the", "lazy", "dog"
    ]
    print(f"\nPerson A's segmented output: {person_a_output}")
    print("(Type: list[str])")
    
    # --- PERSON B CONSUMES IT HERE ---
    # Option 1: MFT Baseline
    mft_tags = pipeline.tag_with_baseline(person_a_output)
    print(f"\nPerson B MFT Tags:    {mft_tags}")
    
    # Option 2: HMM Model
    hmm_tags = pipeline.tag_with_hmm(person_a_output)
    print(f"Person B HMM Tags:    {hmm_tags}")
    
    # Store for downstream: can now feed into morphology or evaluation
    print(f"\nTagged output ready for morphology extension or evaluation:")
    for token, tag in zip(person_a_output, hmm_tags):
        print(f"  {token:15} -> {tag}")


# ============================================================================
# EXAMPLE 3: ERROR ATTRIBUTION (Segmentation vs. Tagging Errors)
# ============================================================================

def example_3_error_attribution():
    """
    Distinguishes segmentation errors (Person A) from tagging errors (Person B).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Error Attribution Analysis")
    print("=" * 70)
    
    train_sents, test_sents = DataLoader.load_brown_corpus()
    pipeline = POSPipeline(language='english')
    pipeline.train_baselines_and_models(train_sents)
    
    # Gold standard from test set
    test_sent = test_sents[0]
    gold_tokens = [w for w, t in test_sent]
    gold_tags = [t for w, t in test_sent]
    
    # Simulate Person A's segmentation (with intentional errors)
    person_a_segmented = gold_tokens.copy()
    # Introduce a deliberate segmentation error
    if len(person_a_segmented) > 3:
        person_a_segmented[2] = "wrong_segment"  # Artificial error
    
    # Person B tags the received output
    person_b_tags = pipeline.tag_with_hmm(person_a_segmented)
    
    print(f"Gold tokens:      {gold_tokens}")
    print(f"Gold tags:        {gold_tags}")
    print(f"\nReceived from A:  {person_a_segmented}")
    print(f"Predicted tags:   {person_b_tags}")
    
    # Error analysis
    evaluator = POSEvaluator()
    error_breakdown = evaluator.attribute_errors(
        person_a_segmented,
        gold_tokens,
        person_b_tags,
        gold_tags
    )
    
    print(f"\n--- Error Attribution ---")
    print(f"Correct:                {error_breakdown['correct']}")
    print(f"Tagging errors (B's):   {error_breakdown['tagging_errors']}")
    print(f"Segmentation errors(A's): {error_breakdown['segmentation_errors']}")
    print(f"Overall accuracy:       {error_breakdown['accuracy']:.4f}")
    print(f"Tagging error rate:     {error_breakdown['tag_error_rate']:.4f}")
    print(f"Segmentation error rate:{error_breakdown['seg_error_rate']:.4f}")


# ============================================================================
# EXAMPLE 4: CONFUSION MATRIX & PER-TAG METRICS
# ============================================================================

def example_4_confusion_matrix():
    """
    Build confusion matrices and per-tag precision/recall.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Confusion Matrix & Per-Tag Analysis")
    print("=" * 70)
    
    train_sents, test_sents = DataLoader.load_brown_corpus()
    pipeline = POSPipeline(language='english')
    pipeline.train_baselines_and_models(train_sents)
    
    # Evaluate on first 50 test sentences
    all_pred = []
    all_gold = []
    
    for sent in test_sents[:50]:
        tokens = [w for w, t in sent]
        tags = [t for w, t in sent]
        pred = pipeline.tag_with_hmm(tokens)
        
        all_pred.extend(pred)
        all_gold.extend(tags)
    
    # Get metrics
    metrics = POSEvaluator.evaluate(all_pred, all_gold)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total predictions: {len(all_pred)}")
    print(f"\nTags evaluated: {metrics['tags']}")
    
    # Show confusion matrix shape
    print(f"Confusion matrix shape: {metrics['confusion_matrix'].shape}")
    print("\nConfusion matrix (first 5x5):")
    print(metrics['confusion_matrix'][:5, :5])


# ============================================================================
# EXAMPLE 5: SPANISH WITH MORPHOLOGY
# ============================================================================

def example_5_spanish_morphology():
    """
    Load UD Spanish-GSD and demonstrate morphology-aware tagging.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Spanish with Morphological Features")
    print("=" * 70)
    
    try:
        datasets = DataLoader.load_ud_spanish()
        
        if 'train' not in datasets:
            print("Warning: Could not load UD Spanish training data")
            return
        
        print(f"\nLoaded datasets: {list(datasets.keys())}")
        
        # Build morphology-aware tagger
        morpho_tagger = MorphologyAwareTagger(language='spanish')
        morpho_tagger.train_from_ud(datasets['train'][:500])  # Train on first 500
        
        # Test on a sample sentence
        test_sent = datasets['test'][0] if 'test' in datasets else datasets['train'][600]
        
        print("\n--- Sample Sentence ---")
        tokens_sp = []
        true_compound = []
        
        for token in test_sent:
            if token['id'] and '-' not in str(token['id']):
                tokens_sp.append(token['form'].lower())
                
                # Extract true compound tag
                upos = token['upos']
                feats = token['feats'] or {}
                gender = feats.get('Gender', ['U'])[0] if feats.get('Gender') else 'U'
                number = feats.get('Number', ['U'])[0] if feats.get('Number') else 'U'
                compound = f"{upos}-{gender}-{number}"
                true_compound.append(compound)
        
        print(f"Tokens:        {tokens_sp}")
        print(f"True compound: {true_compound}")
        
        # Predict morphological tags
        pred_morpho = morpho_tagger.predict(tokens_sp)
        print(f"Predicted:     {pred_morpho}")
        
        # Accuracy on compound tags
        if pred_morpho:
            morpho_acc = sum(p == t for p, t in zip(pred_morpho, true_compound)) / len(true_compound)
            print(f"\nMorphological tag accuracy: {morpho_acc:.4f}")
    
    except Exception as e:
        print(f"Could not run Spanish example: {e}")
        print("Make sure 'conllu' package is installed: pip install conllu")


# ============================================================================
# EXAMPLE 6: PERSON B STANDALONE (Before Integration)
# ============================================================================

def example_6_standalone_testing():
    """
    Person B can work independently using a mock segmenter output
    while waiting for Person A to finish.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Standalone Testing (Mock Segmenter)")
    print("=" * 70)
    
    # Load and train
    train_sents, test_sents = DataLoader.load_brown_corpus()
    pipeline = POSPipeline(language='english')
    pipeline.train_baselines_and_models(train_sents)
    
    # Simulate different segmentation quality levels
    test_sent = test_sents[10]
    gold_tokens = [w for w, t in test_sent]
    
    print(f"Gold tokens ({len(gold_tokens)}): {gold_tokens[:8]}...")
    
    # Scenario 1: Perfect segmentation (Person A got it right)
    perfect_seg = gold_tokens
    perfect_tags = pipeline.tag_with_hmm(perfect_seg)
    print(f"\n[Perfect segmentation] HMM tags: {perfect_tags[:8]}...")
    
    # Scenario 2: Some errors (Person A has bugs)
    with_errors = gold_tokens.copy()
    if len(with_errors) > 2:
        with_errors[1] = "error1"
        with_errors[5] = "error2" if len(with_errors) > 5 else "error2"
    
    error_tags = pipeline.tag_with_hmm(with_errors)
    print(f"\n[Segmentation errors] HMM tags: {error_tags[:8]}...")
    
    # Scenario 3: Evaluate both on gold tags
    gold_tags = [t for w, t in test_sent]
    
    perfect_acc = sum(p == g for p, g in zip(perfect_tags, gold_tags)) / len(gold_tags)
    error_acc = sum(p == g for p, g in zip(error_tags, gold_tags)) / len(gold_tags)
    
    print(f"\nTagging accuracy with perfect segmentation: {perfect_acc:.4f}")
    print(f"Tagging accuracy with segmentation errors:  {error_acc:.4f}")
    print(f"Impact of segmentation on tagging:         {(perfect_acc - error_acc):.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PERSON B: POS TAGGING INTEGRATION EXAMPLES" + " " * 11 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Run all examples
    example_1_mock_data()
    example_2_integration_day()
    example_3_error_attribution()
    example_4_confusion_matrix()
    example_5_spanish_morphology()
    example_6_standalone_testing()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
