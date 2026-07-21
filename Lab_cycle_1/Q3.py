import re

token_pattern = re.compile(r'''
    (?P<abbr>\b(?:[A-Za-z]\.)+[A-Za-z]?\b)    # abbreviations
    | (?P<hyphen>\b\w+(?:-\w+)+\b)            # hyphenated words
    | (?P<contraction>\b\w+n't\b)             # contractions with n't
    | (?P<word>\b\w+\b)                       # plain words
    | (?P<number>\b\d+(?:\.\d+)?\b)           # numbers (including decimals)
    | (?P<punct>[^\w\s])                      # single punctuation/symbol
''', re.VERBOSE)

def tokenize(text):
    tokens = []
    for match in token_pattern.finditer(text):
        token = match.group()
        if token.endswith("n't"):
            base = token [:-3]
            suffix = "n't"
            if base:
                tokens.append(base)
            tokens.append(suffix)
        else:
            tokens.append(token)
    return tokens

sample_text = "I can't believe it's not ice-cream? U.S.A. is great."
result = tokenize(sample_text)
print(result)