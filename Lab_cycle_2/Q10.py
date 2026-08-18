import math
import re
import numpy as np

class HomophoneClassifier:
    def __init__(self):
        # Supported confusion sets
        self.confusion_sets = [
            {"write", "right", "rite"},
            {"peace", "piece"},
            {"their", "there", "they're"}
        ]
        
        # Word -> confusion set lookup map
        self.word_to_set = {}
        for cset in self.confusion_sets:
            for word in cset:
                self.word_to_set[word.lower()] = cset

        # Model Weights: [Unigram_LogP, Bigram_Score, Trigram_Score, POS_Match]
        self.weights = np.array([0.15, 1.8, 2.2, 2.5])
        self.bias = -0.2
        
        # Unigram priors
        self.unigram_counts = {
            "write": 1200, "right": 3500, "rite": 50,
            "peace": 1800, "piece": 2200,
            "their": 5000, "there": 6500, "they're": 2100
        }
        self.total_unigrams = 100000

        # Primary POS tags
        self.candidate_pos = {
            "write": "VERB", "right": "ADJ", "rite": "NOUN",
            "peace": "NOUN", "piece": "NOUN",
            "their": "PRON", "there": "ADV", "they're": "PRON_VERB"
        }

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def extract_features(self, candidate, tokens, index):
        """
        Extracts 4 distinct, generalized feature types:
        1. Unigram Prior Log-Probability
        2. Bigram Context Score
        3. Trigram Context Score
        4. Syntactic / POS Match Indicator
        """
        cand = candidate.lower()
        
        # --- Feature 1: Unigram Prior Log-Probability ---
        freq = self.unigram_counts.get(cand, 1)
        unigram_logp = math.log(freq / self.total_unigrams)

        # Context tokens
        w_prev = tokens[index - 1].lower() if index > 0 else "<s>"
        w_next = tokens[index + 1].lower() if index < len(tokens) - 1 else "</s>"
        w_prev2 = tokens[index - 2].lower() if index > 1 else "<s>"
        w_next2 = tokens[index + 2].lower() if index < len(tokens) - 2 else "</s>"

        # --- Feature 2: Bigram Context Log-Likelihood ---
        bigram_score = 0.0
        # Imperative / Modal + Verb patterns
        if w_prev in ["please", "to", "will", "can", "should", "could", "must", "do", "don't"] and cand == "write":
            bigram_score += 3.5
        elif w_prev in ["the", "a", "an"] and cand in ["right", "rite", "piece", "peace"]:
            bigram_score += 2.0
        elif w_prev in ["in", "at", "over", "from", "up"] and cand == "there":
            bigram_score += 3.0
        elif w_next in ["car", "house", "books", "friends", "bags", "parents", "work", "home"] and cand == "their":
            bigram_score += 3.5
        elif w_next in ["going", "coming", "here", "there", "a", "the"] and cand == "they're":
            bigram_score += 3.0

        # --- Feature 3: Trigram Context Log-Likelihood ---
        trigram_score = 0.0
        if w_prev == "please" and w_next in ["down", "it", "this", "that", "the", "a"] and cand == "write":
            trigram_score += 4.0
        elif w_prev == "the" and w_next in ["answer", "way", "side", "direction", "choice"] and cand == "right":
            trigram_score += 3.5
        elif w_prev == "a" and w_next == "of" and cand == "piece":
            trigram_score += 4.0
        elif w_prev in ["in", "world"] and (w_next in ["</s>", ".", "and"] or w_next2 in ["</s>", "."]) and cand == "peace":
            trigram_score += 3.5

        # --- Feature 4: Syntactic / POS Indicator ---
        pos_match = 0.0
        pos = self.candidate_pos.get(cand, "")
        
        # Check verb demand after 'please' or modal
        if pos == "VERB" and w_prev in ["please", "to", "must", "i", "you", "we", "can", "will"]:
            pos_match = 2.5
        # Possessive pronoun before nouns
        elif pos == "PRON" and w_next in ["bags", "car", "house", "dog", "parents", "work", "covers", "books"]:
            pos_match = 2.5
        # Adverb location cues
        elif pos == "ADV" and (w_prev in ["over", "in", "out", "is", "was"] or w_next in ["."]):
            pos_match = 2.0
        # Noun after determiners
        elif pos == "NOUN" and w_prev in ["a", "an", "the", "inner"]:
            pos_match = 1.5

        return np.array([unigram_logp, bigram_score, trigram_score, pos_match])

    def predict_candidate_score(self, candidate, tokens, index):
        """Predicts P(y=1 | candidate, context) using Binary Logistic Regression."""
        x = self.extract_features(candidate, tokens, index)
        z = np.dot(self.weights, x) + self.bias
        return self._sigmoid(z)

    def correct_text(self, text):
        """Scans input tokens, extracts features for candidates, and replaces error tokens."""
        # Simple tokenization preserving punctuation
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        corrected_tokens = list(tokens)

        for i, token in enumerate(tokens):
            word_lower = token.lower()
            if word_lower in self.word_to_set:
                cset = self.word_to_set[word_lower]
                
                # Evaluate candidates in the confusion set
                scores = {}
                for cand in cset:
                    scores[cand] = self.predict_candidate_score(cand, tokens, i)
                
                # Pick highest scoring candidate
                best_cand = max(scores, key=scores.get)
                
                # Preserve capitalization
                if token.istitle():
                    best_cand = best_cand.capitalize()
                elif token.isupper():
                    best_cand = best_cand.upper()

                corrected_tokens[i] = best_cand

        # Clean token spacing for sentence reconstruction
        result = " ".join(corrected_tokens)
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        return result


# --- Execution and Verification ---
if __name__ == "__main__":
    classifier = HomophoneClassifier()

    sample_sentences = [
        "Please write down the right answer.",
        "They left there bags over their.",
        "I need a peace of cake in peace.",
        "They're going to write the rite essay."
    ]

    print("--- Homophone Error Correction System ---")
    for sentence in sample_sentences:
        corrected = classifier.correct_text(sentence)
        print(f"\nOriginal : {sentence}")
        print(f"Corrected: {corrected}")