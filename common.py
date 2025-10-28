import numpy as np

PATH_PREFIX = './models/8 - lightest/'

def extract_additional_features(texts):
    features = []
    for t in texts:
        emdashes = t.count("—")
        num_chars = len(t)
        num_words = len(t.split())
        num_punct = sum(1 for c in t if c in '.,;:!?()[]{}"\'')
        avg_word_len = num_chars / num_words if num_words > 0 else 0
        features.append([emdashes, num_chars, num_words, num_punct, avg_word_len])
    return np.array(features)

def dummy(text):
    return text
