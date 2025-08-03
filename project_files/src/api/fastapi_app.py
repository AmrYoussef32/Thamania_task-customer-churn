#!/usr/bin/env python3
"""
Customer Churn Prediction API
=============================

Simple FastAPI service to serve the customer churn prediction model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Simple API to predict customer churn using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class CustomerData(BaseModel):
    """Customer data for churn prediction."""

    user_id: str
    total_sessions: int
    total_events: int
    page_diversity: int
    artist_diversity: int
    song_diversity: int
    total_length: float
    avg_song_length: float
    days_active: int
    events_per_session: float
    level: str
    gender: str
    registration: int


class PredictionResponse(BaseModel):
    """Response for churn prediction."""

    user_id: str
    churn_probability: float
    churn_prediction: bool
    confidence: str
    recommendation: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_info: Dict[str, Any]
    timestamp: str


class ModelInfo(BaseModel):
    """Model information."""

    model_name: str
    model_version: str
    training_date: str
    performance_metrics: Dict[str, float]


# Global variables
model = None
preprocessing = None
feature_columns = None
model_info = {}


def load_model():
    """Load the latest trained model."""
    global model, preprocessing, feature_columns, model_info

    try:
        models_dir = "./project_files/models"

        # Check if models directory exists
        if not os.path.exists(models_dir):
            print(f"⚠️ Models directory not found: {models_dir}")
            return False

        # Find latest model files
        model_files = [
            f for f in os.listdir(models_dir) if f.startswith("churn_model_")
        ]
        preprocessing_files = [
            f for f in os.listdir(models_dir) if f.startswith("preprocessing_")
        ]
        feature_files = [
            f for f in os.listdir(models_dir) if f.startswith("feature_columns_")
        ]

        if not model_files:
            print("⚠️ No trained model found in models directory")
            return False

        # Load latest model
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load preprocessing
        if preprocessing_files:
            latest_preprocessing = sorted(preprocessing_files)[-1]
            preprocessing_path = os.path.join(models_dir, latest_preprocessing)
            with open(preprocessing_path, "rb") as f:
                preprocessing = pickle.load(f)

        # Load feature columns
        if feature_files:
            latest_features = sorted(feature_files)[-1]
            feature_path = os.path.join(models_dir, latest_features)
            with open(feature_path, "r") as f:
                feature_columns = json.load(f)

        # Extract model info
        model_info = {
            "model_name": latest_model,
            "model_version": latest_model.split("_")[-1].replace(".pkl", ""),
            "training_date": latest_model.split("_")[2:4],
            "feature_count": len(feature_columns) if feature_columns else 0,
        }

        print(f"✅ Model loaded successfully: {latest_model}")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def preprocess_customer_data(customer_data: CustomerData) -> pd.DataFrame:
    """Preprocess customer data for prediction."""
    try:
        # Create feature dictionary
        features = {
            "total_sessions": customer_data.total_sessions,
            "total_events": customer_data.total_events,
            "page_diversity": customer_data.page_diversity,
            "artist_diversity": customer_data.artist_diversity,
            "song_diversity": customer_data.song_diversity,
            "total_length": customer_data.total_length,
            "avg_song_length": customer_data.avg_song_length,
            "days_active": customer_data.days_active,
            "events_per_session": customer_data.events_per_session,
            "level": customer_data.level,
            "gender": customer_data.gender,
            "registration": customer_data.registration,
        }

        # Create DataFrame
        df = pd.DataFrame([features])

        # Handle categorical features
        categorical_features = ["level", "gender"]
        for feature in categorical_features:
            if feature in df.columns:
                df[feature] = pd.Categorical(df[feature]).codes

        # Handle missing values
        df = df.fillna(0)

        return df

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error preprocessing data: {str(e)}"
        )


def get_confidence_level(probability: float) -> str:
    """Get confidence level based on probability."""
    if probability >= 0.8:
        return "Very High"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    elif probability >= 0.2:
        return "Low"
    else:
        return "Very Low"


def get_recommendation(probability: float, prediction: bool) -> str:
    """Get business recommendation based on prediction."""
    if prediction:
        if probability >= 0.8:
            return "Immediate retention campaign needed - high churn risk"
        elif probability >= 0.6:
            return "Proactive retention campaign recommended"
        else:
            return "Monitor closely and consider retention offers"
    else:
        if probability <= 0.2:
            return "Customer appears loyal - continue current engagement"
        else:
            return "Low risk but monitor for changes in behavior"


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("🚀 Starting Customer Churn Prediction API...")
    if not load_model():
        print("❌ Failed to load model. API may not work correctly.")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    status = "healthy" if model is not None else "unhealthy"

    return HealthResponse(
        status=status,
        model_info=model_info if model_info else {"model_name": "No model loaded"},
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get detailed model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_name=model_info.get("model_name", "Unknown"),
        model_version=model_info.get("model_version", "Unknown"),
        training_date=f"{model_info.get('training_date', ['Unknown'])[0]}_{model_info.get('training_date', ['Unknown'])[1]}",
        performance_metrics={
            "f1_score": 0.778,  # Example metrics
            "auc_score": 0.669,
            "precision": 0.375,
            "recall": 0.667,
        },
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """Predict customer churn."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess customer data
        features_df = preprocess_customer_data(customer_data)

        # Make prediction
        prediction_proba = model.predict_proba(features_df)[0]
        churn_probability = prediction_proba[1]  # Probability of churn
        churn_prediction = model.predict(features_df)[0]

        # Get confidence and recommendation
        confidence = get_confidence_level(churn_probability)
        recommendation = get_recommendation(churn_probability, bool(churn_prediction))

        return PredictionResponse(
            user_id=customer_data.user_id,
            churn_probability=round(churn_probability, 3),
            churn_prediction=bool(churn_prediction),
            confidence=confidence,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", response_model=List[PredictionResponse])
async def predict_churn_batch(customers_data: List[CustomerData]):
    """Predict churn for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []

        for customer_data in customers_data:
            # Preprocess customer data
            features_df = preprocess_customer_data(customer_data)

            # Make prediction
            prediction_proba = model.predict_proba(features_df)[0]
            churn_probability = prediction_proba[1]
            churn_prediction = model.predict(features_df)[0]

            # Get confidence and recommendation
            confidence = get_confidence_level(churn_probability)
            recommendation = get_recommendation(
                churn_probability, bool(churn_prediction)
            )

            predictions.append(
                PredictionResponse(
                    user_id=customer_data.user_id,
                    churn_probability=round(churn_probability, 3),
                    churn_prediction=bool(churn_prediction),
                    confidence=confidence,
                    recommendation=recommendation,
                    timestamp=datetime.now().isoformat(),
                )
            )

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/example-request")
async def get_example_request():
    """Get an example request for testing."""
    return {
        "example_request": {
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
            "registration": 1538352000000,
        },
        "curl_example": 'curl -X POST \'http://localhost:8000/predict\' -H \'Content-Type: application/json\' -d \'{"user_id": "user123", "total_sessions": 15, "total_events": 150, "page_diversity": 8, "artist_diversity": 25, "song_diversity": 45, "total_length": 3600.5, "avg_song_length": 240.0, "days_active": 30, "events_per_session": 10.0, "level": "paid", "gender": "M", "registration": 1538352000000}\'',
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
