from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# ✅ Load model from HuggingFace (NO bert_model folder needed)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.eval()

# Request schema
class NewsRequest(BaseModel):
    text: str

# Health check
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# Prediction function
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
            "confidence": round(conf, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint
@app.post("/predict")
def predict_news(req: NewsRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    return predict(req.text)