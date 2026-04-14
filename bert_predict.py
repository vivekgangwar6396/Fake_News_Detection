from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model + tokenizer
tokenizer = BertTokenizer.from_pretrained("bert_model")
model = BertForSequenceClassification.from_pretrained("bert_model")

model.eval()  # 🔥 important

def predict_news_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():  # 🔥 important
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

    label = "Fake News ⚠" if predicted_class == 0 else "Real News 📰"

    return f"{label} (Confidence: {confidence:.2f})"

# Test
print(predict_news_bert("Stock market hits all-time high"))