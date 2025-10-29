import numpy as np

PATH_PREFIX = './models/02 - sgd/'

EXTRA_FEATURES = True

def extract_additional_features(texts):
    if not EXTRA_FEATURES:
        return np.array([])

    features = []
    for t in texts:
        emdashes = t.count("—")
        num_chars = len(t)
        num_words = len(t.split())
        avg_word_len = num_chars / num_words if num_words > 0 else 0
        features.append([emdashes, avg_word_len])
    return np.array(features)

def dummy(text):
    return text
