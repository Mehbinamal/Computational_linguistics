import math
import random

# ==========================================
# Corpus Definition (Train & Test Split)
# ==========================================
TRAIN_CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "a cat ate the fish",
    "the dog barked at the cat",
    "a dog ate a bone",
    "the cat sleeps on the rug",
    "a dog sleeps on the mat",
    "the fish swims in the water"
]

TEST_CORPUS = [
    "the cat sat on the rug",
    "a dog ate the fish",
    "the dog sleeps on the mat"
]

# Special tokens
BOS = "<s>"  # Beginning of sentence
EOS = "</s>" # End of sentence

class NProblanguageModel:
    def __init__(self, train_corpus):
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.total_unigrams = 0
        self.train_sentences = len(train_corpus)
        
        self._build_models(train_corpus)

    def _tokenize(self, text):
        return text.lower().strip().split()

    def _build_models(self, corpus):
        """a. Compute unsmoothed unigrams and bigrams from train corpus."""
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            
            # Count unigrams (including </s>)
            for word in tokens + [EOS]:
                self.unigram_counts[word] = self.unigram_counts.get(word, 0) + 1
                self.total_unigrams += 1

            # Count bigrams (with <s> ... </s>)
            full_tokens = [BOS] + tokens + [EOS]
            for w1, w2 in zip(full_tokens[:-1], full_tokens[1:]):
                bigram = (w1, w2)
                self.bigram_counts[bigram] = self.bigram_counts.get(bigram, 0) + 1
                
            # Count <s> in unigrams for bigram conditional probability denominator
            self.unigram_counts[BOS] = self.unigram_counts.get(BOS, 0) + self.train_sentences

    # ==========================================
    # Unsmoothed Probabilities
    # ==========================================
    def unigram_prob(self, word):
        """P(w) = count(w) / N"""
        count = self.unigram_counts.get(word, 0)
        return count / self.total_unigrams if self.total_unigrams > 0 else 0.0

    def bigram_prob(self, w1, w2):
        """P(w2 | w1) = count(w1, w2) / count(w1)"""
        count_w1_w2 = self.bigram_counts.get((w1, w2), 0)
        count_w1 = self.unigram_counts.get(w1, 0)
        return count_w1_w2 / count_w1 if count_w1 > 0 else 0.0

    # ==========================================
    # b. Probability of User Text
    # ==========================================
    def compute_sentence_probability(self, text, model_type="bigram"):
        tokens = self._tokenize(text)
        prob = 1.0

        if model_type == "unigram":
            for word in tokens + [EOS]:
                p = self.unigram_prob(word)
                prob *= p
                if prob == 0:
                    break

        elif model_type == "bigram":
            full_tokens = [BOS] + tokens + [EOS]
            for w1, w2 in zip(full_tokens[:-1], full_tokens[1:]):
                p = self.bigram_prob(w1, w2)
                prob *= p
                if prob == 0:
                    break

        return prob

    # ==========================================
    # c. Sentence Generation
    # ==========================================
    def generate_sentence(self, model_type="bigram", max_length=15):
        sentence = []

        if model_type == "unigram":
            # Sample based on unigram distributions (excluding BOS)
            words = [w for w in self.unigram_counts.keys() if w != BOS]
            probs = [self.unigram_prob(w) for w in words]
            # Normalize probabilities
            total_p = sum(probs)
            probs = [p / total_p for p in probs]

            for _ in range(max_length):
                word = random.choices(words, weights=probs)[0]
                if word == EOS:
                    break
                sentence.append(word)

        elif model_type == "bigram":
            current_word = BOS
            for _ in range(max_length):
                # Find all valid bigrams starting with current_word
                candidates = [w2 for (w1, w2) in self.bigram_counts.keys() if w1 == current_word]
                if not candidates:
                    break
                
                probs = [self.bigram_prob(current_word, w2) for w2 in candidates]
                next_word = random.choices(candidates, weights=probs)[0]

                if next_word == EOS:
                    break
                sentence.append(next_word)
                current_word = next_word

        return " ".join(sentence)

    # ==========================================
    # d. Perplexity Calculation
    # ==========================================
    def compute_perplexity(self, corpus, model_type="bigram"):
        """Perplexity = exp( -1/N * sum(log P(w_i)) )"""
        log_prob_sum = 0.0
        total_tokens = 0

        for sentence in corpus:
            tokens = self._tokenize(sentence)

            if model_type == "unigram":
                eval_tokens = tokens + [EOS]
                for word in eval_tokens:
                    p = self.unigram_prob(word)
                    if p == 0:
                        return float('inf')  # Zero probability penalty
                    log_prob_sum += math.log(p)
                    total_tokens += 1

            elif model_type == "bigram":
                full_tokens = [BOS] + tokens + [EOS]
                for w1, w2 in zip(full_tokens[:-1], full_tokens[1:]):
                    p = self.bigram_prob(w1, w2)
                    if p == 0:
                        return float('inf')  # Zero probability penalty
                    log_prob_sum += math.log(p)
                    total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        cross_entropy = -log_prob_sum / total_tokens
        return math.exp(cross_entropy)


# ==========================================
# Main Execution / Interactive Demonstration
# ==========================================
if __name__ == "__main__":
    lm = NProblanguageModel(TRAIN_CORPUS)

    print("==================================================")
    print("      UNSMOOTHED N-GRAM LANGUAGE MODEL")
    print("==================================================")

    # (a) Model Summary
    print(f"\n[a] Trained on {len(TRAIN_CORPUS)} sentences.")
    print(f"    - Vocabulary Size (Unigrams): {len(lm.unigram_counts) - 1}")
    print(f"    - Total Bigram Types: {len(lm.bigram_counts)}")

    # (b) Probability of User Text
    sample_text = "the cat sat on the rug"
    u_prob = lm.compute_sentence_probability(sample_text, "unigram")
    b_prob = lm.compute_sentence_probability(sample_text, "bigram")

    print(f"\n[b] Evaluating Probability for: '{sample_text}'")
    print(f"    - Unigram Model Probability: {u_prob:.10e}")
    print(f"    - Bigram Model Probability : {b_prob:.10e}")

    # (c) Random Sentence Generation
    print("\n[c] Sentence Generation Demonstration:")
    print("    Unigram Generated Sentences:")
    for i in range(3):
        print(f"      {i+1}. {lm.generate_sentence('unigram')}")

    print("    Bigram Generated Sentences:")
    for i in range(3):
        print(f"      {i+1}. {lm.generate_sentence('bigram')}")

    # (d) Perplexity Comparison
    u_ppl = lm.compute_perplexity(TEST_CORPUS, "unigram")
    b_ppl = lm.compute_perplexity(TEST_CORPUS, "bigram")

    print("\n[d] Perplexity on Test Corpus:")
    print(f"    - Unigram Model Perplexity: {u_ppl:.4f}")
    print(f"    - Bigram Model Perplexity : {b_ppl:.4f}")