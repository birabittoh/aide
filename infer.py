import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm

from common import PATH_PREFIX, extract_additional_features, dummy # dummy is used by pickle.load

# ==================== Load Model and Preprocessing ====================
print("Loading model and preprocessing artifacts...")

# Load ensemble model
with open(PATH_PREFIX + 'ensemble_model.pkl', 'rb') as f:
    ensemble = pickle.load(f)
print("✓ Ensemble model loaded")

# Load vectorizer
with open(PATH_PREFIX + 'tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print("✓ TF-IDF vectorizer loaded")

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(PATH_PREFIX + 'bpe_tokenizer')
print("✓ BPE tokenizer loaded")

# Load model info (optional)
try:
    with open('model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    print(f"✓ Model info loaded")
    print(f"  Models: {model_info['models']}")
except:
    print("⚠ Model info not found (optional)")

print("\n" + "="*60)

# ==================== Prediction Functions ====================
def predict_text(text: str, return_proba=True):
    """
    Predict if a text is AI-generated or human-written
    
    Args:
        text (str): Input text to classify
        return_proba (bool): If True, return probability; if False, return binary prediction
    
    Returns:
        float or int: Probability of being AI-generated (0-1) or binary prediction (0/1)
    """
    # Tokenize
    tokenized = tokenizer.tokenize(text)
    
    # Vectorize
    vectorized = vectorizer.transform([tokenized])

    # Feature aggiuntive
    extra_features = extract_additional_features([text])

    # Concatenazione sparse + dense
    full_features = hstack([vectorized, extra_features])
    
    # Predict
    if return_proba:
        proba = ensemble.predict_proba(full_features)[0, 1]
        return proba
    else:
        pred = ensemble.predict(full_features)[0]
        return pred

def predict_batch(texts, show_progress=True):
    """
    Predict multiple texts at once
    
    Args:
        texts (list): List of texts to classify
        show_progress (bool): Show progress bar
    
    Returns:
        pd.DataFrame: DataFrame with texts, predictions, and probabilities
    """
    # Tokenize all texts
    tokenized_texts = []
    iterator = tqdm(texts, desc="Tokenizing") if show_progress else texts
    for text in iterator:
        tokenized_texts.append(tokenizer.tokenize(text))
    
    # Vectorize
    if show_progress:
        print("Vectorizing...")
    vectorized = vectorizer.transform(tokenized_texts)

    # Feature aggiuntive
    extra_features = extract_additional_features(texts)

    # Concatenazione sparse + dense
    full_features = hstack([vectorized, extra_features])

    # Predict
    if show_progress:
        print("Making predictions...")
    probas = ensemble.predict_proba(full_features)[:, 1]
    preds = ensemble.predict(full_features)

    # Create results dataframe
    results = pd.DataFrame({
        'text': texts,
        'prediction': ['AI-generated' if p == 1 else 'Human-written' for p in preds],
        'ai_probability': probas,
        'confidence': np.abs(probas - 0.5) * 2  # 0 = uncertain, 1 = very confident
    })

    return results

def get_model_info():
    """
    Display information about the loaded ensemble model
    """
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Ensemble type: {type(ensemble).__name__}")
    print(f"Number of estimators: {len(ensemble.estimators_)}")
    print("\nBase models:")
    #for i, (name, model) in enumerate(ensemble.estimators):
    #    weight = ensemble.weights[i] if ensemble.weights is not None else 1.0/len(ensemble.estimators_)
    #    print(f"  {i+1}. {name}: {type(model).__name__} (weight: {weight:.2f})")
    #print(f"\nVoting method: {ensemble.voting}")
    print("="*60)

# ==================== Example Usage ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI CONTENT DETECTION - INFERENCE")
    print("="*60)
    
    # Display model info
    get_model_info()
    
    # Example 1: Single text prediction
    print("\n📝 Example 1: Single Text Prediction")
    print("-" * 60)
    
    sample_text = """
    Artificial intelligence — it has revolutionized the way we interact with technology.
    Machine learning algorithms can now process vast amounts of data and identify
    patterns that would be impossible for humans to detect manually.
    """
    
    probability = predict_text(sample_text, return_proba=True)
    prediction = predict_text(sample_text, return_proba=False)
    
    print(f"Text: {sample_text.strip()[:100]}...")
    print(f"\nPrediction: {'AI-generated' if prediction == 1 else 'Human-written'}")
    print(f"AI Probability: {probability:.4f} ({probability*100:.2f}%)")
    print(f"Confidence: {abs(probability - 0.5) * 2:.4f}")
    
    # Example 2: Batch prediction
    print("\n\n📚 Example 2: Batch Prediction")
    print("-" * 60)
    
    sample_texts = [
        """When I do the trick, I cross my left thumb under my right and hold the fire button with it. Then I grip the upper left side of the controller with my right hand and rapidly tap up on the d-pad that way. I usuall turn the controller on it's side too.


Aside from just this, I'ts also a good idea to take note of how mush energy your buster has. I know you said that it wouldn't let you choose your buster parts, but I remember them equipping you with a unit pack alpha, where you got 2 attack, 2 enregy, 2 range, and 1 rapid,
right? Anyway, count the number of shots your energy gives you. If you have no equips given, you will have 3 red shots. If not, 5 green shots. After you shoot your max shots, STOP!
and run backwards. This will solve your problem of moving too close, because not only will you be moving less and shooting faster, you can also run back when not fireing, rather than keep moving forward.

Hope this all helps!(or makes sense ^^;)""",
        """Then nothing is wrong: that's how it works. Maybe it's the only flaw of that trick. To make things simpler for you do this: Once you find an enemy, Hold both B2 (lock on) and Square (shoot), and keep tapping the UP button like mad. So only your left thumb would be hurting at the end of the day instead of both :D

Hmm...no, the Kimotoma Ruins are pretty much the only thing that needs that License to be accessed. The only other thing the license does is making enemies more aggressive.""",
        "*scratches head* No that didn't make much sense.... I do know I shot green shots when I was taking the test. Anyway I am playing on Hard mode and they start you off with an S-Class liscence so I am not worried about it. The only hard fight will be against Glyde's ship when I protect the machine Roll works on. It took me a couple of tries on Normal Mode. I hope I can get past it.",
        "Furthermore, it is important to note that the aforementioned factors contribute significantly to the overall outcome of the analysis.",
        "Hey! How's it going? I'm just chilling at home watching some Netflix."
    ]
    
    results = predict_batch(sample_texts, show_progress=True)
    
    print("\nResults:")
    pd.set_option('display.max_colwidth', 50)
    print(results.to_string(index=False))
    pd.reset_option('display.max_colwidth')
    
    # Example 3: Load and predict from CSV
    print("\n\n📄 Example 3: Predict from CSV (if available)")
    print("-" * 60)
    
    try:
        # Try to load a CSV file
        df = pd.read_csv('test_essays.csv')
        print(f"✓ Found test file with {len(df)} rows")
        
        # Predict on a subset
        sample_size = min(10, len(df))
        sample_df = df.head(sample_size)
        
        print(f"\nProcessing first {sample_size} rows...")
        results = predict_batch(sample_df['text'].tolist(), show_progress=True)
        
        print(f"\nPredictions for first {sample_size} rows:")
        print(results[['prediction', 'ai_probability', 'confidence']].to_string())
        
        # Save results
        results.to_csv('predictions.csv', index=False)
        print(f"\n✓ Full predictions saved to: predictions.csv")
        
    except FileNotFoundError:
        print("⚠ No test_essays.csv found. Skipping this example.")
    except Exception as e:
        print(f"⚠ Error processing CSV: {e}")
    
    print("\n" + "="*60)
