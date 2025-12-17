from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Emotion Classification API", version="1.0.0")

# ===================== КОНФИГУРАЦИЯ =====================
MAX_LENGTH = 50
PADDING = "post"
LABELS = {0: "neutral", 1: "joy", 2: "sadness", 3: "anger", 4: "fear", 5: "surprise"}

# ===================== ПУТИ К МОДЕЛЯМ =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.h5")  # Изменено на .h5
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pickle")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "model_config.json")

model, tokenizer = None, None


@app.on_event("startup")
def load_model():
    global model, tokenizer, MAX_LENGTH, PADDING, LABELS
    
    # Загрузка конфигурации (если есть)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            MAX_LENGTH = config.get('max_length', MAX_LENGTH)
            PADDING = config.get('padding', PADDING)
            if 'class_names' in config:
                LABELS = {int(k): v for k, v in config['class_names'].items()}
        print(f"✓ Конфигурация загружена: max_length={MAX_LENGTH}")
    
    # Загрузка модели
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✓ Модель загружена: {MODEL_PATH}")
        print(f"  TensorFlow version: {tf.__version__}")
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        raise e
    
    # Загрузка токенизатора
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        print(f"✓ Токенизатор загружен: {TOKENIZER_PATH}")
    except Exception as e:
        print(f"✗ Ошибка загрузки токенизатора: {e}")
        raise e


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    emotion: str
    confidence: float


class BatchPredictRequest(BaseModel):
    texts: list[str]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


@app.get("/")
def root():
    return {
        "message": "Emotion Classification API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "labels": LABELS,
        "max_length": MAX_LENGTH
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Предобработка
    seq = tokenizer.texts_to_sequences([request.text])
    pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding=PADDING)
    
    # Предсказание
    pred = model.predict(pad, verbose=0)[0]
    
    predicted_class = int(np.argmax(pred))
    confidence = float(np.max(pred))
    
    return PredictResponse(
        emotion=LABELS.get(predicted_class, f"unknown_{predicted_class}"),
        confidence=round(confidence, 4)
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    # Предобработка всех текстов
    seqs = tokenizer.texts_to_sequences(request.texts)
    pads = pad_sequences(seqs, maxlen=MAX_LENGTH, padding=PADDING)
    
    # Предсказание батчем
    preds = model.predict(pads, verbose=0)
    
    # Формирование ответа
    predictions = []
    for pred in preds:
        predicted_class = int(np.argmax(pred))
        confidence = float(np.max(pred))
        predictions.append(PredictResponse(
            emotion=LABELS.get(predicted_class, f"unknown_{predicted_class}"),
            confidence=round(confidence, 4)
        ))
    
    return BatchPredictResponse(predictions=predictions)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)