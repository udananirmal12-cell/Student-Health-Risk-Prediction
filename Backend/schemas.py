from typing import Literal, Dict
from pydantic import BaseModel, Field


class StudentRecord(BaseModel):
    sleep_duration: float = Field(..., ge=0, le=24, description="Hours of sleep per night")
    heart_rate: float = Field(..., ge=30, le=220, description="Resting heart rate (bpm)")
    bmi: float = Field(..., ge=5, le=80, description="Body Mass Index")
    calorie_expenditure: float = Field(..., ge=0, description="Calories burned per day")
    step_count: float = Field(..., ge=0, description="Steps per day")
    exercise_duration: float = Field(..., ge=0, description="Minutes of exercise per day")
    water_intake: float = Field(..., ge=0, description="Litres of water per day")

    diet_type: Literal["veg", "non-veg", "balanced", "missing"] = "missing"
    stress_level: Literal["low", "medium", "high", "missing"] = "missing"
    sleep_quality: Literal["poor", "average", "good", "missing"] = "missing"
    physical_activity_level: Literal["sedentary", "moderate", "active", "missing"] = "missing"
    smoking_alcohol: Literal["no", "occasional", "yes", "missing"] = "missing"
    gender: Literal["male", "female", "other", "missing"] = "missing"

    class Config:
        json_schema_extra = {
            "example": {
                "sleep_duration": 6.5,
                "heart_rate": 72.0,
                "bmi": 23.5,
                "calorie_expenditure": 2200.0,
                "step_count": 8000.0,
                "exercise_duration": 30.0,
                "water_intake": 2.0,
                "diet_type": "balanced",
                "stress_level": "medium",
                "sleep_quality": "average",
                "physical_activity_level": "moderate",
                "smoking_alcohol": "no",
                "gender": "female",
            }
        }


class PredictionResponse(BaseModel):
    predicted_health_condition: str
    probabilities: Dict[str, float]


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list
