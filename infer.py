import pickle
import numpy as np
from scipy.sparse import hstack
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
import os
import sys
import json
from dotenv import load_dotenv

from common import EXTRA_FEATURES, PATH_PREFIX, extract_additional_features, dummy # dummy is used by pickle.load

# ==================== Load Model and Preprocessing ====================
def load_models(prefix: str = PATH_PREFIX):
    print("Loading model and preprocessing artifacts...")

    # Load model
    with open(prefix + 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("[OK] Model loaded")

    # Load vectorizer
    with open(prefix + 'tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("[OK] TF-IDF vectorizer loaded")

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(prefix + 'bpe_tokenizer')
    print("[OK] BPE tokenizer loaded")

    return model, vectorizer, tokenizer

# ==================== Prediction Functions ====================
# All prediction logic is now handled in predict_batch.

def predict_batch(texts, model, vectorizer, tokenizer, show_progress=True, threshold=0.5):
    """
    Predict multiple texts at once

    Args:
        texts (list): List of texts to classify
        show_progress (bool): Show progress bar
        threshold (float): Probability threshold for classification (default 0.5)

    Returns:
        list: List of dicts with prediction results
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

    if EXTRA_FEATURES:
        # Feature aggiuntive
        extra_features = extract_additional_features(texts)

        # Concatenazione sparse + dense
        vectorized = hstack([vectorized, extra_features])

    # Predict probabilities only
    if show_progress:
        print("Making predictions...")
    probas = model.predict_proba(vectorized)[:, 1]
    preds = (probas > threshold).astype(int)

    results = []
    for i, text in enumerate(texts):
        ai_prob = float(probas[i])
        human_prob = float(1 - ai_prob)
        confidence = float(np.abs(ai_prob - 0.5) * 2)
        results.append({
            'uuid': None,
            'text': text,
            'prediction': 'AI' if preds[i] == 1 else 'Human',
            'confidence': confidence,
            'human_prob': human_prob,
            'ai_prob': ai_prob
        })

    return results

def get_model_info(model):
    """
    Display information about the loaded model
    """
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Model type: {type(model).__name__}")
    print("="*60)

# ==================== Subprocess Mode ====================
def run_batch_prediction(input_path: str, output_path: str):
    """
    Run batch prediction from JSON input file and save to JSON output file.
    Called when script is run as subprocess.
    """
    # Load models
    model, vectorizer, tokenizer = load_models()
    
    # Read input
    with open(input_path, 'r') as f:
        requests = json.load(f)

    texts = [req['text'] for req in requests]
    uuids = [req['uuid'] for req in requests]

    # Use predict_batch for all predictions
    batch_results = predict_batch(texts, model, vectorizer, tokenizer, show_progress=False)

    results = [{**res, 'uuid': uuids[i]} for i, res in enumerate(batch_results)]

    # Write output
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"[OK] Processed {len(results)} requests")

# ==================== Example Usage ====================
if __name__ == "__main__":
    load_dotenv()

    prefix = os.getenv('MODEL_PATH_PREFIX', PATH_PREFIX)
    
    # Check if running in subprocess mode
    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        run_batch_prediction(input_path, output_path)
        sys.exit(0)
    
    # Otherwise run interactive examples
    model, vectorizer, tokenizer = load_models(prefix)
    
    print("\n" + "="*60)
    print("AI CONTENT DETECTION - INFERENCE")
    print("="*60)
    
    # Display model info
    get_model_info(model)
    
    # Example: Batch prediction
    print("\n\nBatch Prediction")
    print("-" * 60)
    
    sample_texts = [
        """Kiao regaz,
questo è il mio primo post su questo fantastiko forum!
Sono unicorniearcobaleni, mi piacciono i gatti, hello kitty, sonic e fare pupazzetti all'uncinetto.
Ho un bellissimo boy, la sua identità è segreta ma intuibile.
Buon forum a tutti,
ci si vede""",
        """When I do the trick, I cross my left thumb under my right and hold the fire button with it. Then I grip the upper left side of the controller with my right hand and rapidly tap up on the d-pad that way. I usuall turn the controller on it's side too.


Aside from just this, I'ts also a good idea to take note of how mush energy your buster has. I know you said that it wouldn't let you choose your buster parts, but I remember them equipping you with a unit pack alpha, where you got 2 attack, 2 enregy, 2 range, and 1 rapid,
right? Anyway, count the number of shots your energy gives you. If you have no equips given, you will have 3 red shots. If not, 5 green shots. After you shoot your max shots, STOP!
and run backwards. This will solve your problem of moving too close, because not only will you be moving less and shooting faster, you can also run back when not fireing, rather than keep moving forward.

Hope this all helps!(or makes sense ^^;)""",
        """Then nothing is wrong: that's how it works. Maybe it's the only flaw of that trick. To make things simpler for you do this: Once you find an enemy, Hold both B2 (lock on) and Square (shoot), and keep tapping the UP button like mad. So only your left thumb would be hurting at the end of the day instead of both :D

Hmm...no, the Kimotoma Ruins are pretty much the only thing that needs that License to be accessed. The only other thing the license does is making enemies more aggressive.""",
        "*scratches head* No that didn't make much sense.... I do know I shot green shots when I was taking the test. Anyway I am playing on Hard mode and they start you off with an S-Class liscence so I am not worried about it. The only hard fight will be against Glyde's ship when I protect the machine Roll works on. It took me a couple of tries on Normal Mode. I hope I can get past it.",
        "Hey! How's it going? I'm just chilling at home watching some Netflix.",
        """The Innovation Powering the Future: Artificial Intelligence

We’re living in one of the most transformative periods in human history — the age of artificial intelligence. What was once science fiction has become an engine of innovation driving progress across every industry imaginable.

From predictive healthcare and autonomous vehicles to creative design and personalized learning, AI is not just automating tasks — it’s amplifying human potential.

The real innovation lies not only in the technology itself, but in how we’re learning to collaborate with it. Humans bring context, empathy, and ethics. AI brings scale, speed, and precision. Together, they form a partnership that redefines what’s possible.

As we continue to explore the frontiers of AI, the key question isn’t just “What can AI do?” but rather “How can we use it responsibly to create a better world?”

The future of innovation is human + machine — and it’s unfolding right now. 🌍💡

#ArtificialIntelligence #Innovation #FutureOfWork #AIRevolution #Technology #Leadership""",
        "Furthermore, it is important to note that the aforementioned factors contribute significantly to the overall outcome of the analysis.",
    ]
    
    results = predict_batch(sample_texts, model, vectorizer, tokenizer, show_progress=True)
    
    print("\nResults:")
    for res in results:
        print(f"text: {res['text'][:40]!r} | prediction: {res['prediction']} | confidence: {res['confidence']:.4f} | human_prob: {res['human_prob']:.4f} | ai_prob: {res['ai_prob']:.4f}")
    print("="*60)
