# LSTM-Text-Prediction

LSTM-based next word prediction system using News Headlines with FastAPI deployment

# LSTM-Based Text Prediction System with FastAPI Deployment

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Accuracy](https://img.shields.io/badge/Accuracy-~20--35%25-brightgreen)

## Group Assignment - 5

### Group Members
| Member | Name | Roll Number |
|--------|------|-------------|
| Member 1 | Nirjara More | 202301100049 |
| Member 2 | Sneha Paliwal | — |
| Member 3 | Vaishnavi Phad | — |
| Member 4 | Disha Satpute | — |

---

## Project Overview

This project implements an LSTM-based next word prediction system trained on real-time **News Headlines** fetched via the NewsAPI.
Given a seed sentence, the model predicts the next N words using deep learning.
The model is deployed as a REST API using FastAPI and tested via Swagger UI.

---

## Architecture

```
NewsAPI Data → Preprocessing → LSTM Model → FastAPI → Swagger UI
```

## Project Structure

```
LSTM-Text-Prediction/
│
├── LSTM_Text_prediction_Ass05.ipynb   # Google Colab notebook
├── lstm_text_model.h5                 # Saved LSTM model (hosted externally)
├── tokenizer.pkl                      # Saved tokenizer
├── training_progress.png              # Training accuracy and loss plots
└── README.md                          # Project documentation
```

---

## Dataset Details

| Field | Details |
|-------|---------|
| Source | NewsAPI (https://newsapi.org) |
| Library Used | `requests` |
| Data Type | News Headlines |
| Topics Covered | 5 categories |
| Data Format | Short textual sequences |

### Topics Collected

1. Technology
2. AI
3. Business
4. Science
5. Health

### Preprocessing Steps

1. Fetch headlines dynamically from NewsAPI across 5 topics (100 articles per topic)
2. Remove duplicate headlines
3. Convert all text to lowercase
4. Remove special characters and digits using regex
5. Remove extra whitespace
6. Tokenize using Keras `Tokenizer`
7. Create input-output sequences (n-gram style)
8. Apply pre-padding to standardize sequence lengths
9. Remove zero-only rows from dataset
10. One-hot encode output labels for classification (or use sparse categorical crossentropy)

---

## LSTM Model Architecture

| Layer | Type | Configuration |
|-------|------|---------------|
| 1 | Embedding | input=vocab_size, output=100 dimensions |
| 2 | LSTM | 128 units |
| 3 | Dense | vocab_size units, activation=softmax |

### Improved Model (Step 7)

| Layer | Type | Configuration |
|-------|------|---------------|
| 1 | Embedding | input=vocab_size, output=100 dimensions |
| 2 | LSTM | 150 units, return_sequences=True |
| 3 | Dropout | rate=0.2 |
| 4 | LSTM | 100 units |
| 5 | Dropout | rate=0.2 |
| 6 | Dense | vocab_size units, activation=softmax |

---

## Mathematical Model of LSTM

**Forget Gate** — decides what information to forget from cell state:

```
f(t) = σ(Wf · [h(t-1), x(t)] + bf)
```

**Input Gate** — decides what new information to store:

```
i(t) = σ(Wi · [h(t-1), x(t)] + bi)
C̃(t) = tanh(Wc · [h(t-1), x(t)] + bc)
```

**Cell State Update** — updates the memory:

```
C(t) = f(t) * C(t-1) + i(t) * C̃(t)
```

**Output Gate** — decides what to output as hidden state:

```
o(t) = σ(Wo · [h(t-1), x(t)] + bo)
h(t) = o(t) * tanh(C(t))
```

**Where:**
- σ = sigmoid activation function
- tanh = hyperbolic tangent activation function
- W = weight matrices
- b = bias vectors
- h(t) = hidden state at time t
- C(t) = cell state at time t
- x(t) = input at time t

---

## Training Results

| Metric | Value |
|--------|-------|
| Initial Epochs | 20 |
| Improved Epochs | 50 |
| Final Retrain Epochs | 60 |
| Batch Size | 512 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Early Stopping | Patience = 5 |
| Learning Rate Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |

---

## Sample Predictions

| Input | Predicted Output |
|-------|-----------------|
| breaking news | breaking news ... (next 5 words predicted) |
| latest technology | latest technology ... (next 5 words predicted) |
| artificial intelligence is | artificial intelligence is ... (next 5 words predicted) |
| stock market today | stock market today ... (next 5 words predicted) |
| global economy is | global economy is ... (next 5 words predicted) |

---

## Deployment — FastAPI

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and available endpoints |
| GET | `/health` | Health check with model and vocab status |
| POST | `/predict` | Predict next words given seed text |

### Sample Request

```json
{
  "seed_text": "artificial intelligence is",
  "num_words": 5
}
```

### Sample Response

```json
{
  "input_text": "artificial intelligence is",
  "predicted_text": "artificial intelligence is likely to improve the",
  "next_words": ["likely", "to", "improve", "the", "world"]
}
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Nirjara06/LSTM_Text_prediction.git
cd LSTM-Text-Prediction
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn tensorflow numpy matplotlib requests pyngrok nest_asyncio
```

### 3. Download pre-trained model

Download from Google Drive links below and place in the project folder.

### 4. Run FastAPI server

```bash
uvicorn prediction_api:app --host 0.0.0.0 --port 8000
```

### 5. Open Swagger UI

```
http://localhost:8000/docs
```

---

## Pre-trained Model Download

Due to GitHub file size limits (25MB), the model is hosted on Google Drive:

- [lstm_text_model.h5 — Google Drive Link](https://drive.google.com/file/d/12DaP4RQKjvh0WSr6MX0RIJkiKUW9IDnv/view?usp=sharing)
- [tokenizer.pkl — Google Drive Link](https://drive.google.com/file/d/1XwdASbANcbxniD-cmUs26FP8NL1FSxNm/view?usp=sharing)

## Google Colab Notebook

Run the full pipeline directly in Google Colab:

- [Open in Google Colab](https://colab.research.google.com/drive/1c7P9074bjKPzbEzrT-JE2gpPxB41-ZQE?usp=sharing)

---

## AI Tool Acknowledgement

| Tool | Purpose | Sections Used |
|------|---------|---------------|
| Claude (Anthropic) | Code guidance and debugging | LSTM architecture, FastAPI setup |

---

## Disclaimer

This is an academic project developed for learning purposes as part of Group Assignment 5.
Not intended for production use.
