import time
import re
from models import SpellingModels
from corrector import SpellingCorrector

# ANSI Color Codes for terminal highlighting
GREEN = '\033[92m'
RESET = '\033[0m'

def run_cli():
    print("Booting up Continuous Terminal CLI Engine...")
    models = SpellingModels()
    corrector = SpellingCorrector(models)
    
    print("="*50)
    print("Live Spelling Corrector Activated.")
    print("Type 'exit' to quit the application.")
    print("="*50)

    while True:
        user_input = input("\nEnter a sentence: ")
        
        if user_input.strip().lower() == 'exit':
            print("Shutting down...")
            break
            
        start_time = time.perf_counter()
        
        # Tokenize preserving basic punctuation
        tokens = re.findall(r"[\w']+|[.,!?;]", user_input)
        corrected_tokens = []
        changes_made = False
        
        for idx, token in enumerate(tokens):
            if not token.isalpha():
                corrected_tokens.append(token)
                continue
                
            clean_token = token.lower()
            
            # Step 1: Check Non-Word Error
            if clean_token not in models.vocab:
                correction = corrector.correct_non_word(clean_token, use_method='B')
            else:
                # Step 2: Check Real-Word Error
                lower_tokens = [t.lower() if t.isalpha() else t for t in tokens]
                correction = corrector.correct_real_word(lower_tokens, idx, use_method='B')
                
            if correction != clean_token:
                # Visually highlight changed words
                corrected_tokens.append(f"{GREEN}**{correction}**{RESET}")
                changes_made = True
            else:
                corrected_tokens.append(token)
                
        # Reconstruct sentence cleanly
        output = " ".join(corrected_tokens)
        # Fix spacing around punctuation
        output = re.sub(r'\s+([.,!?;])', r'\1', output)
        
        latency = (time.perf_counter() - start_time) * 1000 # in ms
        
        print(f"Output: {output}")
        print(f"[Latency: {latency:.2f} ms]")

if __name__ == "__main__":
    run_cli()