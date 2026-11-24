"""FastAPI backend: receives a recording from the front end and returns the recognised sign.

Run (from the backend folder, with the virtual environment active):
    uvicorn main:app --reload --port 8000
Then open http://localhost:8000/docs to try the endpoints in the browser.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's start-up messages

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from preprocess import preprocess

app = FastAPI(title="Hand Gesture Recognition API")

# The browser only lets the front end call this API if its address is allowed here.
# Set ALLOWED_ORIGINS (comma-separated) when the front end runs somewhere else.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

# The trained LSTM, written by train.py. If it doesn't exist yet, the API still
# runs and /predict answers 503. Restart the API after training a new model.
MODELS = Path(__file__).parent / "models"
TOP_GUESSES = 3


def load_model():
    if not (MODELS / "sign_lstm.keras").exists():
        return None, []
    from tensorflow import keras  # imported here: TensorFlow is slow to load
    trained = keras.models.load_model(MODELS / "sign_lstm.keras")
    labels = json.loads((MODELS / "labels.json").read_text(encoding="utf-8"))
    return trained, labels


model, LABELS = load_model()


# --- What the API receives and returns (FastAPI checks these automatically) ---

class Frame(BaseModel):
    t: float | None = None  # ms since the first frame
    left: list[list[float]] | None = None  # 21 x [x, y, z]
    right: list[list[float]] | None = None


class Recording(BaseModel):
    imageWidth: int = Field(gt=0)
    imageHeight: int = Field(gt=0)
    frames: list[Frame] = Field(min_length=1)


class Guess(BaseModel):
    sign: str
    confidence: float  # 0 to 1


class Prediction(BaseModel):
    sign: str
    confidence: float
    top: list[Guess]  # best guesses, highest first


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "modelLoaded": model is not None, "signs": LABELS}


@app.post("/predict", response_model=Prediction)
def predict(recording: Recording):
    try:
        features = preprocess(recording.model_dump())  # shape (60, 126)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Recording received and preprocessed ({features.shape[0]} x {features.shape[1]}), "
            "but no model is trained yet. Run python train.py and restart the API.",
        )

    # The model expects a batch, so features[None] makes it (1, 60, 126).
    # The answer is one probability per sign, adding up to 1.
    probabilities = model(features[None], training=False).numpy()[0]
    best = probabilities.argsort()[::-1][:TOP_GUESSES]  # indexes of the highest first
    top = [Guess(sign=LABELS[i], confidence=float(probabilities[i])) for i in best]
    return Prediction(sign=top[0].sign, confidence=top[0].confidence, top=top)
