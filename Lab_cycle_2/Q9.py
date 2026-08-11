# compare.py
import re
import math
import random
from collections import defaultdict, Counter

# STEP 1: IMPORT your previous Bigram class 
from Q8 import BigramSpellChecker



# STEP 2: Define the Noisy Channel class COMPLETELY FROM SCRATCH
class NoisyChannelSpellChecker:
    def __init__(self):
        self.vocab = set()
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(int)
        self.vocab_size = 0
        self._build_keyboard()

    def _build_keyboard(self):
        rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        self.adjacent = defaultdict(set)
        for row in rows:
            for i, ch in enumerate(row):
                if i > 0:
                    self.adjacent[ch].add(row[i-1])
                if i < len(row) - 1:
                    self.adjacent[ch].add(row[i+1])
        for i in range(len(rows[0])):
            if i < len(rows[1]):
                self.adjacent[rows[0][i]].add(rows[1][i])
                self.adjacent[rows[1][i]].add(rows[0][i])
            if i < len(rows[2]):
                self.adjacent[rows[1][i]].add(rows[2][i])
                self.adjacent[rows[2][i]].add(rows[1][i])

    def tokenize(self, text):
        text = text.lower().replace("'", "")
        return re.findall(r'\b[a-z]+\b', text)

    def train(self, corpus_text):
        tokens = self.tokenize(corpus_text)
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab)
        self.unigram_counts = Counter(tokens)
        for i in range(len(tokens) - 1):
            self.bigram_counts[(tokens[i], tokens[i+1])] += 1
        print(f"[Noisy] Vocab: {self.vocab_size}, Bigrams: {len(self.bigram_counts)}")

    def detect_errors(self, text):
        tokens = self.tokenize(text)
        return [(i, w) for i, w in enumerate(tokens) if w not in self.vocab], tokens

    def score_sentence(self, tokens):
        log_prob = 0.0
        prev = "<s>"
        V = self.vocab_size + 1
        for word in tokens:
            num = self.bigram_counts.get((prev, word), 0) + 1
            denom = 1 + V if prev == "<s>" else self.unigram_counts.get(prev, 0) + V
            log_prob += math.log(num / denom)
            prev = word
        return log_prob

    def candidates_with_ops(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        candidates = {}
        # deletion
        for l, r in splits:
            if r:
                c = l + r[1:]
                if c in self.vocab and c != word:
                    candidates[c] = 'del'
        # transposition
        for l, r in splits:
            if len(r) > 1:
                c = l + r[1] + r[0] + r[2:]
                if c in self.vocab and c != word:
                    candidates[c] = 'trans'
        # substitution
        for l, r in splits:
            if r:
                for ch in letters:
                    c = l + ch + r[1:]
                    if c in self.vocab and c != word:
                        candidates[c] = 'sub'
        # insertion
        for l, r in splits:
            for ch in letters:
                c = l + ch + r
                if c in self.vocab and c != word:
                    candidates[c] = 'ins'
        return candidates

    def error_probability(self, typo, candidate, op):
        """Returns log P(typo | candidate)."""
        if op == 'sub':
            for t, c in zip(typo, candidate):
                if t != c:
                    if c in self.adjacent.get(t, set()):
                        return math.log(0.8)
                    break
            return math.log(0.1)
        elif op == 'trans':
            return math.log(0.6)
        elif op == 'del':
            return math.log(0.2)
        elif op == 'ins':
            return math.log(0.15)
        return 0.0

    def correct_noisy(self, input_text):
        """Noisy Channel correction: LM + Error Model."""
        errors, tokens = self.detect_errors(input_text)
        if not errors:
            return input_text
        corrected = tokens[:]
        for idx, misspelled in errors:
            candidates_dict = self.candidates_with_ops(misspelled)
            if not candidates_dict:
                continue
            best_candidate = None
            best_score = -float('inf')
            for cand, op in candidates_dict.items():
                corrected[idx] = cand
                lm_score = self.score_sentence(corrected)
                err_score = self.error_probability(misspelled, cand, op)
                total = lm_score + err_score
                if total > best_score:
                    best_score = total
                    best_candidate = cand
            corrected[idx] = best_candidate
        return ' '.join(corrected)


# STEP 3: Helper function to generate synthetic typos for benchmarking
def generate_typo(word):
    if len(word) < 3:
        return word
    letters = 'abcdefghijklmnopqrstuvwxyz'
    op = random.choice(['sub', 'del', 'ins', 'trans'])
    if op == 'sub':
        i = random.randint(0, len(word)-1)
        new_ch = random.choice(letters.replace(word[i], ''))
        return word[:i] + new_ch + word[i+1:]
    elif op == 'del':
        i = random.randint(0, len(word)-1)
        return word[:i] + word[i+1:]
    elif op == 'ins':
        i = random.randint(0, len(word))
        return word[:i] + random.choice(letters) + word[i:]
    else:  # trans
        i = random.randint(0, len(word)-2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]


# STEP 4: Benchmarking function (compares both models)
def benchmark(bigram_model, noisy_model, num_tests=100):
    long_words = [w for w in bigram_model.vocab if len(w) >= 4]
    if len(long_words) < num_tests:
        num_tests = len(long_words)

    test_words = random.sample(long_words, num_tests)
    bigram_correct = 0
    noisy_correct = 0

    for word in test_words:
        typo = generate_typo(word)
        while typo in bigram_model.vocab:
            typo = generate_typo(word)

        input_text = f"the {typo}"

        # Bigram correction (from imported class)
        corrected_bigram = bigram_model.correct_sentence(input_text)
        if word in corrected_bigram:
            bigram_correct += 1

        # Noisy correction (from standalone class)
        corrected_noisy = noisy_model.correct_noisy(input_text)
        if word in corrected_noisy:
            noisy_correct += 1

    return bigram_correct / num_tests, noisy_correct / num_tests


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("Loading corpus from 'assets/alice.txt'...")
    with open('assets/alice.txt', 'r', encoding='utf-8') as f:
        corpus = f.read()

    # 1. Train the IMPORTED Bigram model
    bigram_checker = BigramSpellChecker()
    bigram_checker.train(corpus)

    # 2. Train the STANDALONE Noisy model (does not share data with bigram)
    noisy_checker = NoisyChannelSpellChecker()
    noisy_checker.train(corpus)

    # 3. Sample demo
    print("\n" + "="*60)
    sample = "I leve the Cheshire cat."
    print(f"Input: {sample}")
    print("-"*60)

    res_bigram = bigram_checker.correct_sentence(sample)
    res_noisy = noisy_checker.correct_noisy(sample)

    print(f"🔵 Bigram-Only (imported):   {res_bigram}")
    print(f"🟢 Noisy Channel (standalone): {res_noisy}")

    # 4. Quantitative benchmark
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON (100 Synthetic Typos)")
    print("="*60)

    acc_b, acc_n = benchmark(bigram_checker, noisy_checker, num_tests=100)

    print(f"🔵 Bigram-Only Accuracy:   {acc_b * 100:.2f}%")
    print(f"🟢 Noisy Channel Accuracy: {acc_n * 100:.2f}%")
    print(f"📈 Improvement:            {(acc_n - acc_b) * 100:.2f}%")
    print("="*60)