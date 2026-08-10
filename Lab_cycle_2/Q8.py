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
        """
        Bidirectional bigram scoring: for each word, combines
        P(word | prev_word) and P(next_word | word) when both exist.
        Falls back to just P(word | prev_word) at the end of the sentence.
        """
        log_prob = 0.0
        temp_vocab_size = self.vocab_size + 1
        padded = ["<s>"] + tokens + ["</s>"]

        for i in range(1, len(padded) - 1):
            prev_word = padded[i - 1]
            word = padded[i]
            next_word = padded[i + 1]

            # P(word | prev_word)
            num_prev = self.bigram_counts.get((prev_word, word), 0) + 1
            denom_prev = (1 if prev_word == "<s>" else self.unigram_counts.get(prev_word, 0)) + temp_vocab_size
            prob_prev = num_prev / denom_prev

            # P(next_word | word) -- only if next_word isn't the artificial end pad
            if next_word != "</s>":
                num_next = self.bigram_counts.get((word, next_word), 0) + 1
                denom_next = self.unigram_counts.get(word, 0) + temp_vocab_size
                prob_next = num_next / denom_next
                # geometric mean of the two directions in log-space
                log_prob += 0.5 * (math.log(prob_prev) + math.log(prob_next))
            else:
                log_prob += math.log(prob_prev)

        return log_prob

    def correct_sentence(self, input_text):
        errors, tokens = self.detect_errors(input_text)

        if not errors:
            print("\n✅ No spelling errors detected.")
            return input_text
        corrected_tokens = tokens[:]

        for idx, misspelled_word in errors:
            candidates = self.generate_candidates(misspelled_word)
            print(candidates)

            if not candidates:
                print(f"⚠️ No valid candidates found for '{misspelled_word}'. Skipping.")
                continue

            best_candidate = None
            best_score = -float('inf')
            best_freq = -1

            # sorted() makes iteration order deterministic across runs
            for cand in sorted(candidates):
                corrected_tokens[idx] = cand
                score = self.score_sentence(corrected_tokens)
                freq = self.unigram_counts.get(cand, 0)

                # primary key: LM score, tiebreak: raw unigram frequency
                if (score, freq) > (best_score, best_freq):
                    best_score = score
                    best_freq = freq
                    best_candidate = cand

            corrected_tokens[idx] = best_candidate
            print(f"🔍 '{misspelled_word}' -> '{best_candidate}' (Contextual Score: {best_score:.2f}, Freq: {best_freq})")

        return ' '.join(corrected_tokens)

# --- Main Execution / Demo ---
if __name__ == "__main__":
    # 1. Training Corpus
    with open('assets/text.txt', 'r', encoding='utf-8') as f:
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
        