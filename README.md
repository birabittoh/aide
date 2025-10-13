# AIDE - AI Text Detection Service

A memory-efficient Docker service for detecting AI-generated text using the paradetect-deberta-v3-lora model.

## Features

- **Memory Efficient**: Model loaded on-demand and unloaded after processing
- **Queue System**: Handles multiple requests efficiently
- **Deduplication**: Same post ID updates replace previous enqueued requests
- **Callback Support**: Posts results back to your Go backend

## Setup

```bash
# Build the Docker image (uses slim Python base, ~2GB final size)
docker-compose build

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# First run will download models (~1.5GB) to persistent volume
# Subsequent restarts will reuse cached models
```

## API Endpoints

### POST /enqueue

Enqueue a text detection request.

**Request Body:**
```json
{
  "id": "123",
  "content": "Text to analyze...",
  "callback_url": "http://your-backend:8080/callback"
}
```

**Response:**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

### GET /queue

Get current queue status.

**Response:**
```json
{
  "queue_size": 3,
  "is_processing": false,
  "queued_ids": [123, 456, 789]
}
```

## Callback Format

When processing completes, the service will POST to your `callback_url`:

```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "prediction": "AI",
  "confidence": 0.95,
  "human_prob": 0.05,
  "ai_prob": 0.95
}
```

## Go Backend Integration Example

```go
package main

import (
    "bytes"
    "encoding/json"
    "net/http"
    "strconv"
)

type DetectionRequest struct {
    ID          string `json:"id"`
    Content     string `json:"content"`
    CallbackURL string `json:"callback_url"`
}

type EnqueueResponse struct {
    UUID string `json:"uuid"`
}

type DetectionCallback struct {
    UUID       string  `json:"uuid"`
    Prediction string  `json:"prediction"`
    Confidence float64 `json:"confidence"`
    HumanProb  float64 `json:"human_prob"`
    AIProb     float64 `json:"ai_prob"`
}

var pendingUUIDs = map[string]uint{}

func enqueueDetection(postID uint, content string) (string, error) {
    req := DetectionRequest{
        ID:          strconv.Itoa(postID),
        Content:     content,
        CallbackURL: "http://your-backend:8080/api/detection/callback",
    }
    
    body, _ := json.Marshal(req)
    resp, err := http.Post(
        "http://aide:8000/enqueue",
        "application/json",
        bytes.NewBuffer(body),
    )
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()
    
    var result EnqueueResponse
    json.NewDecoder(resp.Body).Decode(&result)

    pendingUUIDs[result.UUID] = postID
    return result.UUID, nil
}

// Callback handler
func detectionCallback(w http.ResponseWriter, r *http.Request) {
    var callback DetectionCallback
    json.NewDecoder(r.Body).Decode(&callback)

    id, ok := pendingUUIDs[callback.UUID]
    if !ok {
      return
    }
    
    // Update your database
    
    w.WriteHeader(http.StatusOK)
}
```

## Development

To run locally without Docker:

```bash
# Install dependencies
uv sync

# Set model cache location (optional)
export TRANSFORMERS_CACHE=~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Run the service
uv run main.py
```

## Troubleshooting

**Model not loading**: First run downloads ~1.5GB. Check logs for download progress.

**Out of memory**: Increase memory limits in docker-compose.yml or ensure model unloading is working.

**Callbacks failing**: Ensure your Go backend is accessible from the Docker container. Use host.docker.internal on Mac/Windows to reach host services.
