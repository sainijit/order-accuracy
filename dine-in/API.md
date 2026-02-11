# Dine-In Order Accuracy API Documentation

## Overview

The Dine-In Order Accuracy API provides REST endpoints for validating restaurant orders against plate images. It includes three main operations:

1. **Validate Single Plate** - Validate one order against a plate image
2. **Validate Batch** - Validate multiple orders with corresponding images
3. **Get Validation Result** - Retrieve validation results by ID

## Base URLs

- **API Base**: `http://localhost:8083/api`
- **Gradio UI**: `http://localhost:7861`
- **API Documentation**: `http://localhost:8083/docs` (Swagger UI)
- **OVMS VLM**: `http://localhost:8002`

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API service is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "dine-in-api",
  "validations_stored": 5
}
```

---

### 2. Validate Single Plate

**POST** `/api/validate`

Validate a single plate image against an order manifest.

**Request:**
- Content-Type: `multipart/form-data`
- **image** (file): Plate image (PNG, JPG, etc.)
- **order** (string): JSON string containing order manifest

**Order JSON Structure:**
```json
{
  "order_id": "order_123",
  "table_number": "5",
  "restaurant": "McDonald's",
  "items": [
    {"name": "Big Mac", "quantity": 1},
    {"name": "French Fries", "quantity": 1},
    {"name": "Coca-Cola", "quantity": 1}
  ],
  "modifiers": ["No pickles"]
}
```

**Response:**
```json
{
  "validation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-09T10:30:45.123456",
  "order_complete": true,
  "missing_items": [],
  "extra_items": [],
  "modifier_validation": {
    "status": "validated",
    "details": []
  },
  "accuracy_score": 0.99,
  "metrics": {
    "end_to_end_latency_ms": 1280,
    "vlm_inference_ms": 840,
    "agent_reconciliation_ms": 290,
    "within_operational_window": true
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/image_01_mcd_combo.png" \
  -F 'order={"order_id":"order_123","table_number":"5","restaurant":"McDonalds","items":[{"name":"Big Mac","quantity":1}]}'
```

**Python Example:**
```python
import requests
import json

url = "http://localhost:8083/api/validate"

order_data = {
    "order_id": "order_123",
    "table_number": "5",
    "restaurant": "McDonald's",
    "items": [
        {"name": "Big Mac", "quantity": 1},
        {"name": "French Fries", "quantity": 1}
    ]
}

files = {'image': open('images/plate.png', 'rb')}
data = {'order': json.dumps(order_data)}

response = requests.post(url, files=files, data=data)
result = response.json()
print(f"Validation ID: {result['validation_id']}")
print(f"Accuracy: {result['accuracy_score']}")
```

---

### 3. Validate Batch

**POST** `/api/validate/batch`

Validate multiple plates with their corresponding orders in a single request.

**Request:**
- Content-Type: `multipart/form-data`
- **images** (files): Multiple plate images
- **orders** (string): JSON array of order manifests (must match image count)

**Orders JSON Structure:**
```json
[
  {
    "image_id": "image_01_mcd_combo",
    "order_id": "order_001",
    "table_number": "5",
    "restaurant": "McDonald's",
    "items": [
      {"name": "Big Mac", "quantity": 1},
      {"name": "French Fries", "quantity": 1}
    ]
  },
  {
    "image_id": "image_02_mcd_variation",
    "order_id": "order_002",
    "table_number": "7",
    "restaurant": "McDonald's",
    "items": [
      {"name": "Chicken Nuggets", "quantity": 1},
      {"name": "Soda", "quantity": 1}
    ]
  }
]
```

**Response:**
```json
{
  "results": [
    {
      "validation_id": "550e8400-e29b-41d4-a716-446655440001",
      "timestamp": "2026-02-09T10:30:45.123456",
      "order_complete": true,
      "missing_items": [],
      "extra_items": [],
      "accuracy_score": 0.99,
      "metrics": {...}
    },
    {
      "validation_id": "550e8400-e29b-41d4-a716-446655440002",
      "timestamp": "2026-02-09T10:30:46.234567",
      "order_complete": false,
      "missing_items": ["French Fries"],
      "extra_items": [],
      "accuracy_score": 0.74,
      "metrics": {...}
    }
  ]
}
```

**Python Example:**
```python
import requests
import json

url = "http://localhost:8083/api/validate/batch"

orders_data = [
    {
        "image_id": "image_01",
        "order_id": "order_001",
        "items": [{"name": "Big Mac", "quantity": 1}]
    },
    {
        "image_id": "image_02",
        "order_id": "order_002",
        "items": [{"name": "Nuggets", "quantity": 1}]
    }
]

files = [
    ('images', open('images/plate1.png', 'rb')),
    ('images', open('images/plate2.png', 'rb'))
]
data = {'orders': json.dumps(orders_data)}

response = requests.post(url, files=files, data=data)
results = response.json()['results']
print(f"Validated {len(results)} orders")
```

---

### 4. Get Validation Result

**GET** `/api/validate/{validation_id}`

Retrieve a specific validation result by its ID.

**Parameters:**
- `validation_id` (path): Unique validation identifier

**Response:**
```json
{
  "validation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-09T10:30:45.123456",
  "order_complete": true,
  "missing_items": [],
  "extra_items": [],
  "modifier_validation": {
    "status": "validated",
    "details": []
  },
  "accuracy_score": 0.99,
  "metrics": {...}
}
```

**cURL Example:**
```bash
curl "http://localhost:8083/api/validate/550e8400-e29b-41d4-a716-446655440000"
```

---

### 5. List All Validations

**GET** `/api/validate`

List all stored validation results.

**Response:**
```json
{
  "total": 10,
  "validations": [
    {
      "validation_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2026-02-09T10:30:45.123456",
      "order_complete": true,
      "accuracy_score": 0.99,
      ...
    },
    ...
  ]
}
```

---

### 6. Delete Validation Result

**DELETE** `/api/validate/{validation_id}`

Delete a validation result by ID.

**Parameters:**
- `validation_id` (path): Unique validation identifier

**Response:**
```json
{
  "message": "Validation 550e8400-e29b-41d4-a716-446655440000 deleted successfully"
}
```

---

## Response Fields

### ValidationResult Schema

| Field | Type | Description |
|-------|------|-------------|
| `validation_id` | string | Unique identifier for this validation |
| `timestamp` | string | ISO 8601 timestamp of validation |
| `order_complete` | boolean | Whether all ordered items are present |
| `missing_items` | array | List of items in order but not on plate |
| `extra_items` | array | List of items on plate but not in order |
| `modifier_validation` | object | Validation status of modifiers |
| `accuracy_score` | float | Overall accuracy score (0.0-1.0) |
| `metrics` | object | Performance metrics |

### Metrics Schema

| Field | Type | Description |
|-------|------|-------------|
| `end_to_end_latency_ms` | int | Total processing time in milliseconds |
| `vlm_inference_ms` | int | VLM model inference time |
| `agent_reconciliation_ms` | int | Order reconciliation time |
| `within_operational_window` | boolean | Whether latency meets SLA (<2000ms) |

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid order JSON: Expecting value: line 1 column 1 (char 0)"
}
```

### 404 Not Found
```json
{
  "detail": "Validation result not found for ID: invalid-id"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Validation error: Image processing failed"
}
```

---

## Testing

Run the included test script:

```bash
python test_api.py
```

Or access the interactive Swagger UI documentation:

```
http://localhost:8083/docs
```

---

## Architecture

The API runs alongside the Gradio UI in the same container:

- **Port 8083**: FastAPI REST endpoints
- **Port 7861**: Gradio web interface
- **Port 8002**: OVMS VLM service

Both services share the same validation logic and can be used independently or together.
