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

        