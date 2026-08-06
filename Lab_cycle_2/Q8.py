import re
import math
from collections import Counter, defaultdict

class BigramSpellChecker:

    def __init__(self):
        self.vocab = set()
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(int)
        self.vocab_size = 0

    # part a Tokenization
    def tokenize(self,text):
        text = text.lower()
        return re.findall(r'\b[a-z]+\b', text)

    def train(self, corpus_text):
        # part a build vocabulary
        tokens = self.tokenize(corpus_text)
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab)

        self.unigram_counts = Counter(tokens)

        # bigram frequency table
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            self.bigram_counts[bigram] += 1

        print("Vocabulary size:", self.vocab_size)
        print(f"Unique bigrams: {len(self.bigram_counts)}")

    # detect misspelled words
    def detect_errors(self, text):
        tokens = self.tokenize(text)
        errors = []
        for idx,token in enumerate(tokens):
            if token not in self.vocab:
                errors.append((idx, token))
        return errors, tokens

    # 1 edit distance candidates
    def generate_candidates(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        candidates = set()

        # 1. Deletion (remove one character)
        for left, right in splits:
            if right:
                candidates.add(left + right[1:])
        
        # 2. Transposition (swap adjacent characters)
        for left, right in splits:
            if len(right) > 1:
                candidates.add(left + right[1] + right[0] + right[2:])
        
        # 3. Substitution (replace one character)
        for left, right in splits:
            if right:
                for c in letters:
                    candidates.add(left + c + right[1:])
        
        # 4. Insertion (add one character)
        for left, right in splits:
            for c in letters:
                candidates.add(left + c + right)
        
        # Filter to keep only words that exist in our vocabulary AND are not the original
        valid_candidates = {c for c in candidates if c in self.vocab and c != word}
        return valid_candidates

    def score_sentence(self, tokens):
        log_prob = 0.0

        prev_word = "<s>"

        temp_vocab_size = self.vocab_size + 1
        for word in tokens:
            # Calculate P(word | prev_word)
            numerator = self.bigram_counts.get((prev_word, word), 0) + 1  # +1 smoothing
            
            # Denominator: count(prev_word) + V
            if prev_word == "<s>":
                # For start token, we don't have a count. We just use a fake count of 1.
                denominator = 1 + temp_vocab_size
            else:
                denominator = self.unigram_counts.get(prev_word, 0) + temp_vocab_size
                
            prob = numerator / denominator
            log_prob += math.log(prob)
            prev_word = word
                
        return log_prob

    def correct_sentence(self, input_text):
        errors, tokens = self.detect_errors(input_text)
        
        if not errors:
            print("\n✅ No spelling errors detected.")
            return input_text
        corrected_tokens = tokens[:]
        
        for idx, misspelled_word in errors:
            candidates = self.generate_candidates(misspelled_word)
            
            if not candidates:
                print(f"⚠️ No valid candidates found for '{misspelled_word}'. Skipping.")
                continue
            
            # Score the sentence for each candidate
            best_candidate = None
            best_score = -float('inf')
            
            # Temporarily replace the misspelled word with the candidate to score
            for cand in candidates:
                corrected_tokens[idx] = cand
                score = self.score_sentence(corrected_tokens)
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            # Update the original token list with the best candidate
            corrected_tokens[idx] = best_candidate
            print(f"🔍 '{misspelled_word}' -> '{best_candidate}' (Contextual Score: {best_score:.2f})")
        
        return ' '.join(corrected_tokens)


# --- Main Execution / Demo ---
if __name__ == "__main__":
    # 1. Training Corpus
    with open('assets/alice.txt', 'r', encoding='utf-8') as f:
        corpus = f.read()

    # 2. Input text with non-word spelling errors
    input_text = "I leve yeu."

    # 3. Initialize and Train
    checker = BigramSpellChecker()
    checker.train(corpus)

    print("\n" + "="*50)
    print(f"Input Text: {input_text}")
    print("="*50)

    # 4. Run the Spell Checker
    corrected_text = checker.correct_sentence(input_text)
    
    print("\n" + "="*50)
    print(f"✅ Corrected Text: {corrected_text}")
    print("="*50)
        