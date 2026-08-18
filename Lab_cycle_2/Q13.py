import numpy as np
from tensorflow.keras.datasets import imdb
from collections import Counter
import math
# --------------------------------------------------
# 1. Load IMDB dataset
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    path="imdb.npz",
    num_words=10000,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))
# --------------------------------------------------
# 2. Naive Bayes classifier
# --------------------------------------------------

class NaiveBayes:

    def __init__(self, k=1):
        self.k = k

        self.word_counts = {
            0: Counter(),
            1: Counter()
        }

        self.total_words = {
            0: 0,
            1: 0
        }

        self.class_counts = {
            0: 0,
            1: 0
        }

        self.vocabulary = set()
    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def fit(self, X, y):

        for document, label in zip(X, y):

            # Count documents in each class
            self.class_counts[label] += 1

            # Count words
            for word in document:

                self.word_counts[label][word] += 1
                self.total_words[label] += 1

                self.vocabulary.add(word)
    # --------------------------------------------------
    # Prior probability P(class)
    # --------------------------------------------------

    def prior_probability(self, label):

        total_documents = sum(
            self.class_counts.values()
        )

        return self.class_counts[label] / total_documents
    # --------------------------------------------------
    # P(word | class)
    #
    # Add-k smoothing
    # --------------------------------------------------

    def word_probability(self, word, label):

        word_count = self.word_counts[label][word]

        vocabulary_size = len(self.vocabulary)

        return (
            (word_count + self.k)
            /
            (
                self.total_words[label]
                + self.k * vocabulary_size
            )
        )
    # --------------------------------------------------
    # Predict one document
    # --------------------------------------------------

    def predict_one(self, document):

        scores = {}

        for label in [0, 1]:

            # Start with log prior
            score = math.log(
                self.prior_probability(label)
            )

            # Add log likelihood for every word
            for word in document:

                probability = self.word_probability(
                    word,
                    label
                )

                score += math.log(probability)

            scores[label] = score

        # Return class with highest probability
        return max(
            scores,
            key=scores.get
        )
    # --------------------------------------------------
    # Predict multiple documents
    # --------------------------------------------------

    def predict(self, X):

        predictions = []

        for document in X:

            predictions.append(
                self.predict_one(document)
            )

        return np.array(predictions)
# --------------------------------------------------
# 3. Train and test for different k values
# --------------------------------------------------

k_values = [0.25, 0.75, 1]

results = []

for k in k_values:

    print("\n==============================")
    print("Training with k =", k)
    print("==============================")

    model = NaiveBayes(k=k)

    model.fit(x_train, y_train)

    # Predict
    predictions = model.predict(x_test)

    # Accuracy
    accuracy = np.mean(
        predictions == y_test
    )

    print("Accuracy:", accuracy)

    results.append(
        (k, accuracy)
    )
# --------------------------------------------------
# 4. Display comparison
# --------------------------------------------------
print("\n\n===== RESULTS =====")

for k, accuracy in results:

    print(
        f"k = {k:<4} | "
        f"Accuracy = {accuracy:.6f}"
    )