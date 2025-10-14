from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DebertaV2ForSequenceClassification, DebertaV2TokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import PeftModel
from pydantic import BaseModel, HttpUrl
import httpx
import asyncio
import gc
import torch
from uuid import UUID, uuid4
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Global queue and processing flag
request_queue: dict[UUID, "DetectionRequest"] = {}
processing_lock = asyncio.Lock()
is_processing = False

# Config from environment variables
tokenizer_name = os.getenv("TOKENIZER_NAME", "srikanthgali/paradetect-deberta-v3-lora")
base_model_name = os.getenv("BASE_MODEL_NAME", "microsoft/deberta-v3-large")
model_name = os.getenv("MODEL_NAME", "srikanthgali/paradetect-deberta-v3-lora")
callback_timeout = int(os.getenv("CALLBACK_TIMEOUT", "30"))
queue_check_interval = int(os.getenv("QUEUE_CHECK_INTERVAL", "5"))
truncation = os.getenv("TRUNCATION", "True").lower() in ("true", "1", "yes", "on")
padding = os.getenv("PADDING", "True").lower() in ("true", "1", "yes", "on")
max_length = int(os.getenv("MAX_LENGTH", "512"))
host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8000"))

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

def load_model() -> tuple[DebertaV2TokenizerFast, DebertaV2ForSequenceClassification, PeftModel]:
    """Load model on-demand"""
    tokenizer: DebertaV2TokenizerFast = AutoTokenizer.from_pretrained(tokenizer_name)
    base_model: DebertaV2ForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_name)
    model.eval()
    
    return tokenizer, base_model, model

def unload_model(tokenizer: DebertaV2TokenizerFast, base_model: DebertaV2ForSequenceClassification, model: PeftModel):
    """Unload model to free memory"""
    del tokenizer
    del base_model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict_text_origin(text: str, tokenizer: DebertaV2TokenizerFast, model: PeftModel, uuid: UUID) -> CallbackPayload:
    """Run prediction on text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    
    with torch.no_grad():
        outputs: SequenceClassifierOutput = model(**inputs)
        if outputs.logits is None:
            raise ValueError("Model output logits are None")

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
                tokenizer, base_model, model = load_model()
                
                # Process each request
                for req_uuid, req in items:
                    try:
                        # Run prediction
                        result = predict_text_origin(req.content, tokenizer, model, req_uuid)
                        
                        # Send callback
                        async with httpx.AsyncClient(timeout=callback_timeout) as client:
                            await client.post(
                                str(req.callback_url),
                                json=result.model_dump(mode='json')
                            )
                    except Exception as e:
                        print(f"Error processing request {req_uuid}: {e}")
                
                # Unload model to free memory
                unload_model(tokenizer, base_model, model)
                
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
async def queue_status() -> QueueStatus:
    """Get queue status"""
    return QueueStatus(
        is_processing=is_processing,
        queued_ids=[req.id for req in request_queue.values()]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=host, port=port)
