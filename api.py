from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging
from fastapi.middleware.cors import CORSMiddleware

# ================= INIT =================
app = FastAPI(title="Fake News Detection API")

# Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)

# ================= LOAD MODEL =================
try:
    tokenizer = BertTokenizer.from_pretrained("bert_model")
    model = BertForSequenceClassification.from_pretrained("bert_model")
    model.eval()
    logging.info("✅ Model loaded successfully")
except Exception as e:
    logging.error(f"❌ Model loading failed: {e}")
    raise e

# ================= REQUEST SCHEMA =================
class NewsRequest(BaseModel):
    text: str

# ================= HEALTH CHECK =================
@app.get("/")
def health_check():
    return {"status": "API is running 🚀"}

# ================= PREDICTION =================
def predict(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        conf = probs[0][pred].item()

        label = "Fake" if pred == 0 else "Real"

        return {
            "label": label,
            "confidence": round(conf, 3),
            "status": "success"
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# ================= API ENDPOINT =================
@app.post("/predict")
def predict_news(req: NewsRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    return predict(req.text)