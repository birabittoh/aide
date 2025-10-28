import sys
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import httpx
import asyncio
from uuid import UUID, uuid4
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import json
import tempfile

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


async def run_infer_subprocess(requests: list[tuple[str, str]]) -> dict:
    """
    Run infer.py in a subprocess and get predictions.
    Memory is freed when subprocess terminates.
    """
    # Create temporary files for input and output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        input_path = input_file.name
        json.dump([{'uuid': uuid, 'text': text} for uuid, text in requests], input_file)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        # Run infer.py as subprocess
        process = await asyncio.create_subprocess_exec(
            sys.executable, 'infer.py', input_path, output_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise Exception(f"infer.py failed: {error_msg}")
        
        # Read results
        with open(output_path, 'r') as f:
            results = json.load(f)
        
        return results
        
    finally:
        # Cleanup temp files
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass


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
                # Prepare data for subprocess
                requests_data = [(str(uuid), req.content) for uuid, req in items]
                requests_map = {str(uuid): req for uuid, req in items}
                
                # Run predictions in subprocess
                results = await run_infer_subprocess(requests_data)
                
                # Send callbacks
                for result in results:
                    try:
                        req_uuid_str = result['uuid']
                        req = requests_map[req_uuid_str]
                        callback_url = str(req.callback_url) if req.callback_url else f"http://localhost:{port}/debug"
                        
                        callback_payload = {
                            'uuid': req_uuid_str,
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'human_prob': result['human_prob'],
                            'ai_prob': result['ai_prob']
                        }
                        
                        async with httpx.AsyncClient(timeout=callback_timeout) as client:
                            await client.post(callback_url, json=callback_payload)
                            
                    except Exception as e:
                        print(f"Error sending callback for {req_uuid_str}: {e}")
                
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
