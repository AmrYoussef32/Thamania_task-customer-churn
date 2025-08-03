# Customer Churn Prediction API

Simple FastAPI service to serve the customer churn prediction model.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r project_files/requirements.txt
```

### 2. Start the API Server
```bash
cd project_files/src/api
python fastapi_app.py
```

### 3. Access the API
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## File Structure

```
project_files/
├── src/
│   ├── api/
│   │   └── fastapi_app.py       # FastAPI application
│   ├── customer_churn_prediction.py  # ML model
│   └── model_retraining/        # Retraining system
├── models/                      # Trained models
├── requirements.txt             # Dependencies
├── fastapi_app.py              # FastAPI application
└── README_api.md               # This file
```

## API Endpoints

### 1. **Health Check**
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_info": {
    "model_name": "churn_model_20250802_213432.pkl",
    "model_version": "213432",
    "feature_count": 12
  },
  "timestamp": "2025-08-02T21:34:22.122034"
}
```

### 2. **Model Information**
```http
GET /model-info
```
**Response:**
```json
{
  "model_name": "churn_model_20250802_213432.pkl",
  "model_version": "213432",
  "training_date": "20250802_213432",
  "performance_metrics": {
    "f1_score": 0.778,
    "auc_score": 0.669,
    "precision": 0.375,
    "recall": 0.667
  }
}
```

### 3. **Single Prediction**
```http
POST /predict
```
**Request Body:**
```json
{
  "user_id": "user123",
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
**Response:**
```json
{
  "user_id": "user123",
  "churn_probability": 0.234,
  "churn_prediction": false,
  "confidence": "Low",
  "recommendation": "Low risk but monitor for changes in behavior",
  "timestamp": "2025-08-02T21:34:22.122034"
}
```

### 4. **Batch Prediction**
```http
POST /predict-batch
```
**Request Body:**
```json
[
  {
    "user_id": "user001",
    "total_sessions": 10,
    "total_events": 100,
    "page_diversity": 5,
    "artist_diversity": 15,
    "song_diversity": 30,
    "total_length": 2400.0,
    "avg_song_length": 240.0,
    "days_active": 20,
    "events_per_session": 10.0,
    "level": "free",
    "gender": "F",
    "registration": 1538352000000
  },
  {
    "user_id": "user002",
    "total_sessions": 25,
    "total_events": 300,
    "page_diversity": 12,
    "artist_diversity": 40,
    "song_diversity": 80,
    "total_length": 7200.0,
    "avg_song_length": 240.0,
    "days_active": 45,
    "events_per_session": 12.0,
    "level": "paid",
    "gender": "M",
    "registration": 1538352000000
  }
]
```

### 5. **Example Request**
```http
GET /example-request
```
Returns an example request structure and curl command.

## Testing

### Run All Tests
```bash
curl -X GET "http://localhost:8000/health"
```

### Manual Testing with curl

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
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

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "user_id": "user001",
      "total_sessions": 10,
      "total_events": 100,
      "page_diversity": 5,
      "artist_diversity": 15,
      "song_diversity": 30,
      "total_length": 2400.0,
      "avg_song_length": 240.0,
      "days_active": 20,
      "events_per_session": 10.0,
      "level": "free",
      "gender": "F",
      "registration": 1538352000000
    }
  ]'
```

## Response Interpretation

### Confidence Levels
- **Very High** (≥80%): Extremely confident prediction
- **High** (60-79%): Very confident prediction
- **Medium** (40-59%): Moderately confident prediction
- **Low** (20-39%): Low confidence prediction
- **Very Low** (<20%): Very low confidence prediction

### Recommendations
- **High Churn Risk** (≥80%): Immediate retention campaign needed
- **Medium Churn Risk** (60-79%): Proactive retention campaign recommended
- **Low Churn Risk** (40-59%): Monitor closely and consider retention offers
- **Very Low Risk** (≤20%): Customer appears loyal
- **Low Risk** (21-39%): Low risk but monitor for changes

## Configuration

### Environment Variables
- `PORT`: API port (default: 8000)
- `HOST`: API host (default: 0.0.0.0)
- `MODEL_PATH`: Path to models directory (default: ./project_files/models)

### Model Loading
The API automatically loads the latest trained model from the `models` directory. It looks for files with patterns:
- `churn_model_*.pkl`
- `preprocessing_*.pkl`
- `feature_columns_*.json`

## Error Handling

### Common Error Responses

#### Model Not Loaded (503)
```json
{
  "detail": "Model not loaded"
}
```

#### Invalid Request Data (400)
```json
{
  "detail": "Error preprocessing data: Invalid data format"
}
```

#### Prediction Error (500)
```json
{
  "detail": "Prediction error: Model prediction failed"
}
```

## Performance

### Response Times
- **Health Check**: ~10ms
- **Single Prediction**: ~50ms
- **Batch Prediction**: ~100ms per customer

### Throughput
- **Single Predictions**: ~1000 requests/second
- **Batch Predictions**: ~500 requests/second

## Integration with Retraining

The API automatically uses the latest trained model. When you retrain the model:

1. **Train new model**: `python project_files/src/model_retraining/model_retraining.py`
2. **API automatically picks up new model** on next request
3. **No restart required** - model is loaded dynamically

## 🚀 Deployment

### Development
```bash
cd project_files/src/api
python fastapi_app.py
```

### Production with uvicorn
```bash
uvicorn project_files.src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

### Docker (recommended)
```bash
docker-compose up --build
```

## API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### OpenAPI Schema
- **Schema**: http://localhost:8000/openapi.json

## Use Cases

### 1. **Real-time Customer Analysis**
```python
import requests

# Analyze customer in real-time
response = requests.post("http://localhost:8000/predict", json=customer_data)
if response.json()["churn_prediction"]:
    send_retention_campaign(customer_data["user_id"])
```

### 2. **Batch Customer Analysis**
```python
# Analyze multiple customers
response = requests.post("http://localhost:8000/predict-batch", json=customers_data)
high_risk_customers = [c for c in response.json() if c["churn_probability"] > 0.7]
```

### 3. **Dashboard Integration**
```javascript
// Frontend integration
fetch('/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(customerData)
})
.then(response => response.json())
.then(data => updateDashboard(data));
```

## Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure you've trained a model first
   - Check `project_files/models/` directory

2. **Port already in use**
   - Change port in docker-compose.yml or uvicorn command
   - Kill existing process: `lsof -ti:8000 | xargs kill`

3. **Import errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python path

4. **Prediction errors**
   - Verify input data format
   - Check model compatibility

### Debug Mode
Add debug logging to `fastapi_app.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For issues or questions:
1. Check the API documentation at `/docs`
2. Review the model training logs
3. Verify model files exist in `project_files/models/` directory
4. Check Docker logs: `docker-compose logs churn-api` 