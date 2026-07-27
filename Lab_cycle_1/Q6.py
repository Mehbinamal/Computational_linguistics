from collections import defaultdict

#corpus
raw_corpus = [
    "the quick brown fox jumps over the lazy dog",
    "i love natural language processing and machine learning",
    "tokenization is a crucial step for text preprocessing",
    "we use byte pair encoding to learn subword tokens"
]

#preprocessing
words = []
for line in raw_corpus:
    for word in line.lower().split():
        words.append(list (word)  + ['</w>'])  # Append end-of-word token

def print_corpus(words):
    out = []
    for i,w in enumerate(words):
        out.append(' '.join(w))
    return ' | '.join(out)

print("Initial corpus (first 4 words):")
print(print_corpus(words))
print(f"Initial vocabulary size: {len(set(tok for w in words for tok in w))}")
print("-" * 60)

# BPE loop
def get_pair_freqs(words):
    freqs = defaultdict(int)
    for word in words:
        for i in range(len(word) - 1):
            freqs[(word[i],word[i+1])] += 1
    return freqs

def merge_pair(words, pair):
    # Ensure both tokens are plain strings (defensive)
    tok1 = str(pair[0]) if not isinstance(pair[0], str) else pair[0]
    tok2 = str(pair[1]) if not isinstance(pair[1], str) else pair[1]
    new_token = tok1 + tok2   # e.g. 'e' + '</w>' -> 'e</w>'

    new_words = []
    for word in words:
        # Safety: make sure every token in this word is a string
        clean_word = [str(t) for t in word]
        new_word = []
        i = 0
        while i < len(clean_word):
            # Look for an exact match of the pair
            if i < len(clean_word) - 1 and clean_word[i] == tok1 and clean_word[i+1] == tok2:
                new_word.append(new_token)   # <--- APPEND, never extend or overwrite
                i += 2
            else:
                new_word.append(clean_word[i])
                i += 1
        new_words.append(new_word)   # <--- APPEND the word, never flatten
    return new_words

num_merges = 0
max_merges = 50

while num_merges < max_merges:
    freqs = get_pair_freqs(words)
    if not freqs:
        break
    # Find the most frequent pair
    best_pair = max(freqs, key=freqs.get)
    best_freq = freqs[best_pair]

    # Stop if everything appears only once (no useful merges)
    if best_freq == 1:
        break

    # Perform the merge
    words = merge_pair(words, best_pair)
    new_token = ''.join(best_pair)

    # Show intermediate step
    vocab = set(tok for w in words for tok in w)
    print(f"Merge #{num_merges + 1}:")
    print(f"  Pair: {best_pair} -> '{new_token}' (frequency = {best_freq})")
    print(f"  Current vocabulary size: {len(vocab)}")
    print(f"  Corpus sample (first 4 words): {print_corpus(words)}")
    print("-" * 60)

    num_merges += 1

