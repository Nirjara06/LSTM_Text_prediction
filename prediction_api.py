
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize app
app = FastAPI(
    title="LSTM News Prediction API",
    description="Predicts next words from news headlines using LSTM",
    version="1.0.0"
)

# Load model and tokenizer
model = load_model("lstm_text_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# IMPORTANT: use same value as training
SEQ_LENGTH = 23   # replace with your max_seq_len-1
VOCAB_SIZE = len(tokenizer.word_index) + 1

# Request body
class PredictRequest(BaseModel):
    seed_text: str
    num_words: int = 5

# Response body
class PredictResponse(BaseModel):
    input_text: str
    predicted_text: str
    next_words: list

# Root endpoint
@app.get("/")
def home():
    return {
        "message": "News LSTM Prediction API is running!",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        }
    }

# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "LSTM",
        "vocab_size": VOCAB_SIZE
    }

# Prediction logic
def predict_next_words(seed_text, num_words):
    result = seed_text
    next_words = []

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([result])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=SEQ_LENGTH,
            padding="pre"
        )

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = int(np.argmax(predicted_probs))

        predicted_word = tokenizer.index_word.get(predicted_index, "")

        next_words.append(predicted_word)
        result += " " + predicted_word

    return result, next_words

# /predict endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    predicted_text, next_words = predict_next_words(
        request.seed_text,
        request.num_words
    )

    return PredictResponse(
        input_text=request.seed_text,
        predicted_text=predicted_text,
        next_words=next_words
    )
