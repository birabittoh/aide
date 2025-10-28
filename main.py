from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import httpx
import asyncio
import gc
import pickle
from scipy.sparse import hstack
from transformers import PreTrainedTokenizerFast
from uuid import UUID, uuid4
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

from common import extract_additional_features, dummy # dummy is used by pickle.load

# Load environment variables from .env file
load_dotenv()

# Global queue and processing flag
request_queue: dict[UUID, "EnqueueRequest"] = {}
processing_lock = asyncio.Lock()
is_processing = False

# Config from environment variables
model_path = os.getenv("MODEL_PATH", "./models/8 - lightest/")
callback_timeout = int(os.getenv("CALLBACK_TIMEOUT", "30"))
queue_check_interval = int(os.getenv("QUEUE_CHECK_INTERVAL", "5"))
host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8000"))

class EnqueueRequest(BaseModel):
    id: str
    content: str
    callback_url: Optional[HttpUrl]

class EnqueueResponse(BaseModel):
    uuid: UUID

class CallbackPayload(BaseModel):
    uuid: UUID | None
    prediction: str
    confidence: float
    human_prob: float
    ai_prob: float

class QueueStatus(BaseModel):
    is_processing: bool
    queued_ids: list[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background processor
    asyncio.create_task(process_queue())
    yield
    # Shutdown: cleanup
    pass

app = FastAPI(lifespan=lifespan)

def load_model():
    """Load ensemble model, vectorizer, and tokenizer on-demand"""
    # Load ensemble model
    with open(model_path + 'ensemble_model.pkl', 'rb') as f:
        ensemble = pickle.load(f)
    
    # Load vectorizer
    with open(model_path + 'tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path + 'bpe_tokenizer')
    
    return ensemble, vectorizer, tokenizer

def unload_model(ensemble, vectorizer, tokenizer):
    """Unload model to free memory"""
    del ensemble
    del vectorizer
    del tokenizer
    gc.collect()

def predict_text_origin(text: str, ensemble, vectorizer, tokenizer, uuid: UUID) -> CallbackPayload:
    """Run prediction on text using ensemble model"""
    # Tokenize
    tokenized = tokenizer.tokenize(text)
    
    # Vectorize
    vectorized = vectorizer.transform([tokenized])
    
    # Extract additional features
    extra_features = extract_additional_features([text])
    
    # Concatenate sparse + dense features
    full_features = hstack([vectorized, extra_features])
    
    # Predict
    probabilities = ensemble.predict_proba(full_features)[0]
    prediction = ensemble.predict(full_features)[0]
    
    human_prob = probabilities[0]
    ai_prob = probabilities[1]
    
    return CallbackPayload(
        uuid=uuid,
        prediction="AI" if prediction == 1 else "Human",
        confidence=max(human_prob, ai_prob),
        human_prob=human_prob,
        ai_prob=ai_prob
    )

async def process_queue():
    """Background task to process queued requests"""
    global is_processing
    
    while True:
        await asyncio.sleep(queue_check_interval)
        
        if not request_queue:
            continue
            
        async with processing_lock:
            if is_processing:
                continue
                
            is_processing = True
            
            # Get all current requests
            items = list(request_queue.items())
            request_queue.clear()
            
            try:
                # Load model once for batch
                ensemble, vectorizer, tokenizer = load_model()
                
                # Process each request
                for req_uuid, req in items:
                    try:
                        # Run prediction
                        result = predict_text_origin(req.content, ensemble, vectorizer, tokenizer, req_uuid)

                        callback_url = str(req.callback_url) if req.callback_url else f"http://localhost:{port}/debug"
                        
                        # Send callback
                        async with httpx.AsyncClient(timeout=callback_timeout) as client:
                            await client.post(
                                callback_url,
                                json=result.model_dump(mode='json')
                            )
                    except Exception as e:
                        print(f"Error processing request {req_uuid}: {e}")
                
                # Unload model to free memory
                unload_model(ensemble, vectorizer, tokenizer)
                
            except Exception as e:
                print(f"Error in queue processing: {e}")
            finally:
                is_processing = False

@app.post("/debug")
async def debug_endpoint(request: CallbackPayload):
    """Debug endpoint to inspect incoming requests"""
    print(request.model_dump(mode='json'))
    return {"status": "ok"}

@app.post("/enqueue", response_model=EnqueueResponse)
async def enqueue_request(request: EnqueueRequest):
    """Enqueue a detection request"""
    # Check if request with same id already exists
    existing_uuid = None
    for uuid, req in list(request_queue.items()):
        if req.id == request.id:
            existing_uuid = uuid
            del request_queue[uuid]
            break
    
    # Use existing UUID or create new one
    req_uuid = existing_uuid if existing_uuid else uuid4()
    request_queue[req_uuid] = request
    
    return EnqueueResponse(uuid=req_uuid)

@app.get("/queue")
async def queue_status() -> QueueStatus:
    """Get queue status"""
    return QueueStatus(
        is_processing=is_processing,
        queued_ids=[req.id for req in request_queue.values()]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=host, port=port)