# 🧠 AI Fake News Detection System (BERT + FastAPI + Streamlit)

## 📌 Overview

This project is an AI-powered Fake News Detection System that classifies news as **Real** or **Fake** using a deep learning model (**BERT**).

It is built using a **full-stack architecture**:

* 🎨 Frontend: Streamlit
* ⚙️ Backend: FastAPI
* 🧠 Model: BERT (Transformers)
* 🌐 External API: NewsAPI (for live news)

---

## 🚀 Features

* 🧠 BERT-based deep learning model
* ⚡ FastAPI backend for prediction
* 🎨 Streamlit interactive UI
* 🌐 Live news analysis (News API)
* 📊 Confidence score & analytics
* 📜 History tracking
* 🔍 Explainable AI output

---

## 📁 Project Structure

```
Fake_News_Detection/
│
├── bert_model/          # Trained BERT model
├── api.py               # FastAPI backend
├── app.py               # Streamlit frontend
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🛠️ Installation

### 🔹 1. Clone Repository

```
git clone <your-repo-link>
cd Fake_News_Detection
```

---

### 🔹 2. Create Virtual Environment

#### Windows:

```
python -m venv venv310
venv310\Scripts\activate
```

---

### 🔹 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 🔹 4. Install Additional NLP Data

```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

### 🔹 5. Add News API Key

Replace in `app.py`:

```
YOUR_API_KEY
```

Get API key from:
👉 https://newsapi.org/

---

## ▶️ How to Run

### 🟢 Step 1: Start Backend (FastAPI)

```
uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### 🟢 Step 2: Run Frontend (Streamlit)

Open new terminal:

```
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

## 🔄 Workflow

```
User Input → Streamlit UI → FastAPI → BERT Model → Prediction → UI
```

---

## 🧪 Example

Input:

```
Stock market hits all-time high
```

Output:

```
Real News 📰 (Confidence: 0.89)
```

---

## 📊 Technologies Used

* Python
* BERT (Transformers)
* FastAPI
* Streamlit
* PyTorch
* Pandas
* Requests

---

## 🎯 Advantages

* High accuracy using deep learning
* Real-time prediction
* Scalable architecture
* User-friendly interface

---

## ⚠️ Limitations

* Requires computational resources
* Depends on dataset quality
* Needs internet for API

---

## 🔮 Future Scope

* Cloud deployment
* Multilingual support
* Voice input
* Improved explainability

---

## 🧠 Author

* Name: *Your Name*
* Project: Final Year Major Project

---

## 📜 License

This project is for educational purposes.


pip freeze > requirements.txt
---
