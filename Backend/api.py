import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from schemas import StudentRecord, PredictionResponse, HealthCheckResponse
from predictor import get_predictor

app = FastAPI(
    title="Student Health Risk Prediction API",
    description="Serves predictions from a trained XGBoost model that classifies "
                 "a student's health condition as fit / at-risk / unhealthy.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_startup_error = None


@app.on_event("startup")
def load_model_on_startup():
    # Load (and cache) the model + artifacts once when the server boots,
    # instead of on the first request, so failures show up immediately in
    # the terminal instead of on someone's first click in the UI.
    global _startup_error
    try:
        get_predictor()
        print("Model + preprocessing artifacts loaded successfully.")
    except Exception as e:
        _startup_error = str(e)
        print(f"WARNING: model failed to load at startup:\n{e}")


@app.get("/", tags=["meta"])
def root():
    return {"message": "Student Health Risk Prediction API. See /docs for usage."}


@app.get("/health", response_model=HealthCheckResponse, tags=["meta"])
def health_check():
    try:
        predictor = get_predictor()
        return HealthCheckResponse(
            status="ok",
            model_loaded=True,
            classes=predictor.class_labels,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(record: StudentRecord):
    """Predict the health condition for a single student record."""
    try:
        predictor = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")

    result = predictor.predict_single(record.model_dump())
    return PredictionResponse(**result)


@app.post("/predict_csv", tags=["prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV (e.g. the competition's test.csv) and get back a CSV with
    predictions + class probabilities appended to every row.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    try:
        predictor = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")

    result_df = predictor.predict_batch(df)

    buffer = io.StringIO()
    result_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
