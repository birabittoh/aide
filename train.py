import gc
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from common import PATH_PREFIX, dummy, extract_additional_features

from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

# Set random state
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==================== Loading Data ====================
print("Loading data...")
org_train = pd.read_csv('train_essays.csv')
train = pd.read_csv("train_v2_drcat_02.csv", sep=',')

# ==================== Remove Duplicate Rows ====================
print("Removing duplicates...")
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

# ==================== Remove Near-Duplicates ====================
print("Detecting and removing near-duplicates...")

from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVec
from sklearn.metrics.pairwise import cosine_similarity

# Quick TF-IDF for duplicate detection (word-level, simpler)
dedup_vectorizer = TfidfVec(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

# Vectorize texts
print("  Vectorizing texts for similarity check...")
dedup_vectors = dedup_vectorizer.fit_transform(train['text'])

# Find near-duplicates using cosine similarity
print("  Computing similarities...")
similarity_threshold = 0.95  # 95% similar = near duplicate

# Keep track of indices to remove
indices_to_remove = set()

# Process in batches to avoid memory issues
batch_size = 1000
n_samples = len(train)

for i in range(0, n_samples, batch_size):
    end_idx = min(i + batch_size, n_samples)
    batch_vectors = dedup_vectors[i:end_idx]
    
    # Compare with all subsequent samples
    if end_idx < n_samples:
        remaining_vectors = dedup_vectors[end_idx:]
        similarities = cosine_similarity(batch_vectors, remaining_vectors)
        
        # Find pairs above threshold
        high_sim = np.where(similarities > similarity_threshold)
        
        for row, col in zip(high_sim[0], high_sim[1]):
            # Mark the later index for removal
            indices_to_remove.add(end_idx + col)
    
    if i % 5000 == 0:
        print(f"    Processed {i}/{n_samples} samples, found {len(indices_to_remove)} duplicates so far")

print(f"  Found {len(indices_to_remove)} near-duplicate samples")

# Remove near-duplicates
if len(indices_to_remove) > 0:
    indices_to_keep = [i for i in range(len(train)) if i not in indices_to_remove]
    train = train.iloc[indices_to_keep].reset_index(drop=True)
    print(f"  Removed {len(indices_to_remove)} near-duplicates")
    print(f"  Dataset size after deduplication: {len(train)}")
else:
    print("  No near-duplicates found")

del dedup_vectorizer, dedup_vectors
gc.collect()

# ==================== Split Train/Test ====================
print("Splitting train/test sets...")
# Create train/test split with indices to maintain consistency
train_idx, test_idx = train_test_split(
    range(len(train)), 
    test_size=0.2, 
    random_state=RANDOM_STATE, 
    stratify=train['label'].values
)

train_data = train.iloc[train_idx].reset_index(drop=True)
test_data = train.iloc[test_idx].reset_index(drop=True)

X_train = train_data['text'].values
y_train = train_data['label'].values
X_test = test_data['text'].values
y_test = test_data['label'].values

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train labels: {len(y_train)}, Test labels: {len(y_test)}")

# ==================== Configuration Parameters ====================
LOWERCASE = False
VOCAB_SIZE = 30522

# ==================== Byte-Pair Encoding Tokenizer Training ====================
print("Training BPE tokenizer...")
# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

# Train tokenizer on test set
dataset_test = Dataset.from_dict({'text': list(X_test)})
def train_corp_iter(): 
    for i in range(0, len(dataset_test), 1000):
        yield dataset_test[i : i + 1000]["text"]

raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

print("Tokenizing texts...")
tokenized_texts_test = []
for text in tqdm(list(X_test), desc="Tokenizing test"):
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = []
for text in tqdm(list(X_train), desc="Tokenizing train"):
    tokenized_texts_train.append(tokenizer.tokenize(text))

# ==================== TF-IDF Vectorization ====================
print("Vectorizing with TF-IDF...")
def dummy(text):
    return text

vectorizer = TfidfVectorizer(
    ngram_range=(3, 5), 
    lowercase=False, 
    sublinear_tf=True, 
    analyzer='word',
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None, 
    strip_accents='unicode'
)

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_

vectorizer = TfidfVectorizer(
    ngram_range=(3, 5), 
    lowercase=False, 
    sublinear_tf=True, 
    vocabulary=vocab,
    analyzer='word',
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None, 
    strip_accents='unicode'
)

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

print(f"TF-IDF shape - Train: {tf_train.shape}, Test: {tf_test.shape}")

# Estrazione delle feature extra
print("Adding handcrafted features...")
extra_train = extract_additional_features(X_train)
extra_test = extract_additional_features(X_test)

# Concatenazione con le matrici TF-IDF
tf_train = hstack([tf_train, extra_train])
tf_test = hstack([tf_test, extra_test])

print(f"New feature shape - Train: {tf_train.shape}, Test: {tf_test.shape}")
print(f"Labels shape - y_train: {y_train.shape}, y_test: {y_test.shape}")

# Verify alignment
assert tf_train.shape[0] == len(y_train), f"Train data mismatch: {tf_train.shape[0]} vs {len(y_train)}"
assert tf_test.shape[0] == len(y_test), f"Test data mismatch: {tf_test.shape[0]} vs {len(y_test)}"
print("✓ Data and labels are properly aligned")

# ==================== Model Training and Prediction ====================
print("Training ensemble model...")

# MultinomialNB
clf = MultinomialNB(alpha=0.02)

# SGD Classifier
sgd_model = SGDClassifier(
    max_iter=8000, 
    tol=1e-4, 
    loss="modified_huber", 
    random_state=RANDOM_STATE
)

# Use StackingClassifier
ensemble = StackingClassifier(
    estimators=[
        ('mnb', clf),
        ('sgd', sgd_model),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    passthrough=False,
    n_jobs=-1
)

print("\nTraining ensemble model (this may take a while)...")
ensemble.fit(tf_train, y_train)
gc.collect()

print("\nMaking predictions...")
y_pred_proba = ensemble.predict_proba(tf_test)[:, 1]
y_pred = ensemble.predict(tf_test)

# ==================== Calculate Metrics ====================
print("Calculating metrics...")

roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {
    'metric': ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'value': [roc_auc, accuracy, precision, recall, f1]
}

metrics_df = pd.DataFrame(metrics)
print("\n" + "="*50)
print("MODEL METRICS")
print("="*50)
print(metrics_df.to_string(index=False))
print("="*50)

# ==================== Export Model and Metrics ====================
print("\nExporting model and metrics...")

# Ensure path exists
os.makedirs(PATH_PREFIX, exist_ok=True)

# Export the ensemble model
with open(PATH_PREFIX + 'ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
print("✓ Model exported to: ensemble_model.pkl")

# Export the vectorizer
with open(PATH_PREFIX + 'tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ Vectorizer exported to: tfidf_vectorizer.pkl")

# Export the tokenizer
tokenizer.save_pretrained(PATH_PREFIX + 'bpe_tokenizer')
print("✓ Tokenizer exported to: bpe_tokenizer/")

# Export metrics to CSV
metrics_df.to_csv(PATH_PREFIX + 'model_metrics.csv', index=False)
print("✓ Metrics exported to: model_metrics.csv")

# Export model info
model_info = {
    'models': ['MultinomialNB', 'SGDClassifier'],
    'train_size': len(X_train),
    'test_size': len(X_test),
    'random_state': RANDOM_STATE,
    'vocab_size': VOCAB_SIZE
}

with open(PATH_PREFIX + 'model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ Model info exported to: model_info.pkl")

print("\n✅ Training complete! All artifacts saved.")
