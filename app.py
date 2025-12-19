"""
FastAPI application for Medical Appointment No-Show Prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import pickle
import numpy as np
import uvicorn
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Medical Appointment No-Show Prediction API",
    description="Predicts whether a patient will show up for their medical appointment",
    version="1.0.0"
)

# Load the trained model and feature names
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("Model loaded successfully!")
    print(f"Expected features: {feature_names}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = None

# Define input schema
class PredictionInput(BaseModel):
    Gender: int  # 0 for Female, 1 for Male
    Age: int
    Scholarship: int  # 0 or 1
    Hipertension: int  # 0 or 1
    Diabetes: int  # 0 or 1
    Alcoholism: int  # 0 or 1
    Handcap: int  # 0-4
    SMS_received: int  # 0 or 1
    days_in_advance: int
    month_appointment_5: float = 0.0
    month_appointment_6: float = 0.0
    day_of_week_appointment_1: float = 0.0
    day_of_week_appointment_2: float = 0.0
    day_of_week_appointment_3: float = 0.0
    day_of_week_appointment_4: float = 0.0
    day_of_week_appointment_5: float = 0.0
    number_of_previous_apptms: float = 0.0
    number_of_previous_noshows: float = 0.0

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Gender": 1,
                "Age": 30,
                "Scholarship": 0,
                "Hipertension": 0,
                "Diabetes": 0,
                "Alcoholism": 0,
                "Handcap": 0,
                "SMS_received": 1,
                "days_in_advance": 14,
                "month_appointment_5": 1.0,
                "month_appointment_6": 0.0,
                "day_of_week_appointment_1": 0.0,
                "day_of_week_appointment_2": 1.0,
                "day_of_week_appointment_3": 0.0,
                "day_of_week_appointment_4": 0.0,
                "day_of_week_appointment_5": 0.0,
                "number_of_previous_apptms": 2.0,
                "number_of_previous_noshows": 0.0
            }
        }
    )

# Define output schema
class PredictionOutput(BaseModel):
    prediction: int  # 0 = will show, 1 = will not show
    probability: float
    message: str

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Medical Appointment No-Show Prediction API",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/healthz": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/healthz")
def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "features_count": len(feature_names) if feature_names else 0
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Predict whether a patient will show up for their appointment
    
    Returns:
    - prediction: 0 (will show) or 1 (will not show)
    - probability: probability of no-show
    - message: human-readable prediction
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to array in the correct order
        features = [
            input_data.Gender,
            input_data.Age,
            input_data.Scholarship,
            input_data.Hipertension,
            input_data.Diabetes,
            input_data.Alcoholism,
            input_data.Handcap,
            input_data.SMS_received,
            input_data.days_in_advance,
            input_data.month_appointment_5,
            input_data.month_appointment_6,
            input_data.day_of_week_appointment_1,
            input_data.day_of_week_appointment_2,
            input_data.day_of_week_appointment_3,
            input_data.day_of_week_appointment_4,
            input_data.day_of_week_appointment_5,
            input_data.number_of_previous_apptms,
            input_data.number_of_previous_noshows
        ]
        
        # Reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = int(model.predict(features_array)[0])
        probability = float(model.predict_proba(features_array)[0][1])
        
        # Create message
        if prediction == 0:
            message = f"Patient will likely SHOW UP (confidence: {(1-probability)*100:.1f}%)"
        else:
            message = f"Patient will likely NOT SHOW (confidence: {probability*100:.1f}%)"
        
        return PredictionOutput(
            prediction=prediction,
            probability=probability,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
