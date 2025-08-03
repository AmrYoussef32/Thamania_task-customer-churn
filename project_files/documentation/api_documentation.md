# Customer Churn Prediction API Documentation

This document provides comprehensive documentation for the Customer Churn Prediction API, including endpoint descriptions, request/response formats, and usage examples.

## 🚀 API Overview

The Customer Churn Prediction API is built using FastAPI and provides RESTful endpoints for predicting customer churn probability. The API serves the trained machine learning model and includes health monitoring capabilities.

**Base URL**: `http://localhost:8000`
**API Version**: v1.0
**Framework**: FastAPI
**Documentation**: Interactive docs available at `/docs`

## 📋 API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Returns the health status of the API and model information.

**Response Format**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_info": {
    "model_name": "gradient_boosting_churn_model.pkl",
    "model_version": "1.0.0",
    "last_updated": "2024-01-15T09:00:00Z",
    "performance_metrics": {
      "f1_score": 0.778,
      "accuracy": 0.812,
      "auc": 0.891
    }
  }
}
```

**Example Request**:
```bash
curl -X GET "http://localhost:8000/health"
```

### 2. Predict Churn

**Endpoint**: `POST /predict`

**Description**: Predicts the churn probability for a given customer based on their features.

**Request Format**:
```json
{
  "user_id": "customer_001",
  "total_sessions": 15,
  "total_events": 150,
  "page_diversity": 8,
  "artist_diversity": 25,
  "song_diversity": 45,
  "total_length": 3600.5,
  "avg_song_length": 240.0,
  "days_active": 30,
  "events_per_session": 10.0,
  "level": "paid",
  "gender": "M",
  "registration": 1538352000000
}
```

**Response Format**:
```json
{
  "user_id": "customer_001",
  "churn_probability": 0.234,
  "churn_prediction": false,
  "confidence": 0.766,
  "risk_level": "low",
  "features_used": [
    "total_sessions",
    "total_events",
    "page_diversity",
    "artist_diversity",
    "song_diversity",
    "total_length",
    "avg_song_length",
    "days_active",
    "events_per_session",
    "level",
    "gender"
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "customer_001",
    "total_sessions": 15,
    "total_events": 150,
    "page_diversity": 8,
    "artist_diversity": 25,
    "song_diversity": 45,
    "total_length": 3600.5,
    "avg_song_length": 240.0,
    "days_active": 30,
    "events_per_session": 10.0,
    "level": "paid",
    "gender": "M",
    "registration": 1538352000000
  }'
```

### 3. Batch Predictions

**Endpoint**: `POST /predict/batch`

**Description**: Predicts churn probability for multiple customers in a single request.

**Request Format**:
```json
{
  "customers": [
    {
      "user_id": "customer_001",
      "total_sessions": 15,
      "total_events": 150,
      "page_diversity": 8,
      "artist_diversity": 25,
      "song_diversity": 45,
      "total_length": 3600.5,
      "avg_song_length": 240.0,
      "days_active": 30,
      "events_per_session": 10.0,
      "level": "paid",
      "gender": "M",
      "registration": 1538352000000
    },
    {
      "user_id": "customer_002",
      "total_sessions": 5,
      "total_events": 50,
      "page_diversity": 3,
      "artist_diversity": 10,
      "song_diversity": 15,
      "total_length": 1200.0,
      "avg_song_length": 180.0,
      "days_active": 10,
      "events_per_session": 10.0,
      "level": "free",
      "gender": "F",
      "registration": 1538352000000
    }
  ]
}
```

**Response Format**:
```json
{
  "predictions": [
    {
      "user_id": "customer_001",
      "churn_probability": 0.234,
      "churn_prediction": false,
      "confidence": 0.766,
      "risk_level": "low"
    },
    {
      "user_id": "customer_002",
      "churn_probability": 0.789,
      "churn_prediction": true,
      "confidence": 0.789,
      "risk_level": "high"
    }
  ],
  "batch_size": 2,
  "processing_time_ms": 45,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 📊 Data Models

### Customer Features

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `user_id` | string | Unique customer identifier | "customer_001" |
| `total_sessions` | integer | Total number of user sessions | 15 |
| `total_events` | integer | Total user interactions/events | 150 |
| `page_diversity` | integer | Number of unique pages visited | 8 |
| `artist_diversity` | integer | Number of unique artists listened to | 25 |
| `song_diversity` | integer | Number of unique songs played | 45 |
| `total_length` | float | Total listening time in seconds | 3600.5 |
| `avg_song_length` | float | Average song length in seconds | 240.0 |
| `days_active` | integer | Number of days user was active | 30 |
| `events_per_session` | float | Average events per session | 10.0 |
| `level` | string | Subscription level ("free" or "paid") | "paid" |
| `gender` | string | User gender ("M" or "F") | "M" |
| `registration` | integer | Registration timestamp (milliseconds) | 1538352000000 |

### Prediction Response

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `user_id` | string | Customer identifier | "customer_001" |
| `churn_probability` | float | Probability of churn (0-1) | 0.234 |
| `churn_prediction` | boolean | Binary churn prediction | false |
| `confidence` | float | Model confidence in prediction | 0.766 |
| `risk_level` | string | Risk category ("low", "medium", "high") | "low" |
| `features_used` | array | List of features used for prediction | ["total_sessions", ...] |
| `timestamp` | string | Prediction timestamp | "2024-01-15T10:30:00Z" |

## 🎯 Risk Levels

The API categorizes churn risk into three levels:

- **Low Risk** (0-0.3): Low probability of churn
- **Medium Risk** (0.3-0.7): Moderate probability of churn
- **High Risk** (0.7-1.0): High probability of churn

## 🔧 Error Handling

### Common Error Responses

#### 400 Bad Request
```json
{
  "detail": "Invalid input data",
  "errors": [
    {
      "field": "total_sessions",
      "message": "Value must be positive"
    }
  ]
}
```

#### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "total_sessions"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Model prediction failed",
  "error": "Model file not found"
}
```

## 🚀 Getting Started

### 1. Start the API Server

```bash
# Using uv
uv run python project_files/src/api/fastapi_app.py

# Using pip
cd project_files/src/api
python fastapi_app.py
```

### 2. Access Interactive Documentation

Open your browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @customer_data.json
```

## 📈 Performance Metrics

### API Performance
- **Response Time**: ~50ms per prediction
- **Throughput**: ~1000 requests/second
- **Model Loading**: ~2 seconds on startup
- **Memory Usage**: ~500MB for model and dependencies

### Model Performance
- **F1 Score**: 0.778
- **Accuracy**: 0.812
- **ROC AUC**: 0.891
- **Precision**: 0.789
- **Recall**: 0.768

## 🔒 Security Considerations

### Input Validation
- All input fields are validated using Pydantic models
- Type checking and range validation
- Required field validation

### Rate Limiting
- Consider implementing rate limiting for production
- Monitor API usage and performance
- Set appropriate request limits

### Data Privacy
- No customer data is stored permanently
- Predictions are not logged by default
- Consider implementing data anonymization

## 🛠️ Development and Testing

### Running Tests

```bash
# Run API tests
uv run pytest project_files/tests/test_api.py

# Run with coverage
uv run pytest --cov=project_files/src/api
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -p customer_data.json -T application/json http://localhost:8000/predict

# Using wrk
wrk -t12 -c400 -d30s --script=post.lua http://localhost:8000/predict
```

### Monitoring

```bash
# Check API health
curl http://localhost:8000/health

# Monitor logs
tail -f api.log
```

## 📝 Example Usage in Different Languages

### Python

```python
import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000"

# Customer data
customer_data = {
    "user_id": "customer_001",
    "total_sessions": 15,
    "total_events": 150,
    "page_diversity": 8,
    "artist_diversity": 25,
    "song_diversity": 45,
    "total_length": 3600.5,
    "avg_song_length": 240.0,
    "days_active": 30,
    "events_per_session": 10.0,
    "level": "paid",
    "gender": "M",
    "registration": 1538352000000
}

# Make prediction
response = requests.post(
    f"{API_BASE_URL}/predict",
    json=customer_data,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    prediction = response.json()
    print(f"Churn Probability: {prediction['churn_probability']:.3f}")
    print(f"Risk Level: {prediction['risk_level']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### JavaScript

```javascript
// Customer data
const customerData = {
    user_id: "customer_001",
    total_sessions: 15,
    total_events: 150,
    page_diversity: 8,
    artist_diversity: 25,
    song_diversity: 45,
    total_length: 3600.5,
    avg_song_length: 240.0,
    days_active: 30,
    events_per_session: 10.0,
    level: "paid",
    gender: "M",
    registration: 1538352000000
};

// Make prediction
fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(customerData)
})
.then(response => response.json())
.then(data => {
    console.log(`Churn Probability: ${data.churn_probability.toFixed(3)}`);
    console.log(`Risk Level: ${data.risk_level}`);
})
.catch(error => {
    console.error('Error:', error);
});
```

### R

```r
library(httr)
library(jsonlite)

# Customer data
customer_data <- list(
    user_id = "customer_001",
    total_sessions = 15,
    total_events = 150,
    page_diversity = 8,
    artist_diversity = 25,
    song_diversity = 45,
    total_length = 3600.5,
    avg_song_length = 240.0,
    days_active = 30,
    events_per_session = 10.0,
    level = "paid",
    gender = "M",
    registration = 1538352000000
)

# Make prediction
response <- POST(
    "http://localhost:8000/predict",
    body = toJSON(customer_data, auto_unbox = TRUE),
    content_type("application/json")
)

if (response$status_code == 200) {
    prediction <- fromJSON(rawToChar(response$content))
    cat(sprintf("Churn Probability: %.3f\n", prediction$churn_probability))
    cat(sprintf("Risk Level: %s\n", prediction$risk_level))
} else {
    cat("Error:", response$status_code, "\n")
}
```

## 🔄 API Versioning

The API supports versioning through URL paths:

- **Current Version**: `/predict` (v1.0)
- **Versioned Endpoints**: `/v1/predict` (future)
- **Backward Compatibility**: Maintained for 6 months

## 📞 Support and Contact

For API support and questions:
- **Documentation**: Check `/docs` for interactive documentation
- **Issues**: Report bugs through the project repository
- **Questions**: Contact the development team

## 📄 License

This API is part of the Customer Churn Prediction project and is provided for educational and demonstration purposes. 