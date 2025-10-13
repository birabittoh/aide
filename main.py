from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel, HttpUrl
import httpx
import asyncio
import gc
import torch
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

# Global queue and processing flag
request_queue: dict[UUID, "DetectionRequest"] = {}
processing_lock = asyncio.Lock()
is_processing = False

class DetectionRequest(BaseModel):
    id: str
    content: str
    callback_url: HttpUrl

class EnqueueResponse(BaseModel):
    uuid: UUID

class CallbackPayload(BaseModel):
    uuid: UUID | None
    prediction: str
    confidence: float
    human_prob: float
    ai_prob: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background processor
    asyncio.create_task(process_queue())
    yield
    # Shutdown: cleanup
    pass

app = FastAPI(lifespan=lifespan)

def load_model():
    """Load model on-demand"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained("srikanthgali/paradetect-deberta-v3-lora")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large",
        num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, "srikanthgali/paradetect-deberta-v3-lora")
    model.eval()
    
    return tokenizer, model

def unload_model(tokenizer, model):
    """Unload model to free memory"""
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict_text_origin(text: str, tokenizer, model: PeftModel, uuid: UUID) -> CallbackPayload:
    """Run prediction on text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
        
        human_prob = probabilities[0][0].item()
        ai_prob = probabilities[0][1].item()

        return CallbackPayload(
            uuid=uuid,
            prediction="AI" if prediction.item() == 1 else "Human",
            confidence=max(human_prob, ai_prob),
            human_prob=human_prob,
            ai_prob=ai_prob
        )

async def process_queue():
    """Background task to process queued requests"""
    global is_processing
    
    while True:
        await asyncio.sleep(1)  # Check queue every second
        
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
                tokenizer, model = load_model()
                
                # Process each request
                for req_uuid, req in items:
                    try:
                        # Run prediction
                        result = predict_text_origin(req.content, tokenizer, model, req_uuid)
                        
                        # Send callback
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            await client.post(
                                str(req.callback_url),
                                json=result.model_dump(mode='json')
                            )
                    except Exception as e:
                        print(f"Error processing request {req_uuid}: {e}")
                
                # Unload model to free memory
                unload_model(tokenizer, model)
                
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
async def enqueue_request(request: DetectionRequest):
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
async def queue_status():
    """Get queue status"""
    return {
        "queue_size": len(request_queue),
        "is_processing": is_processing,
        "queued_ids": [req.id for req in request_queue.values()]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
