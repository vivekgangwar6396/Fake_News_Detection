import streamlit as st
import requests
import pandas as pd

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🧠",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: white;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-text {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}
.glass {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.result-box {
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- API STATUS ----------------
def check_api():
    try:
        res = requests.get(API_URL, timeout=2)
        return res.status_code == 200
    except:
        return False

# ---------------- API CALL ----------------
def predict_news_api(text):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=5
        )

        if response.status_code != 200:
            st.error("⚠ API error occurred")
            return None, None

        data = response.json()
        return data["label"], data["confidence"]

    except requests.exceptions.ConnectionError:
        st.error("🚨 API not running! Start FastAPI first.")
        return None, None

    except requests.exceptions.Timeout:
        st.error("⏳ API timeout. Try again.")
        return None, None

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None, None

# ---------------- LIVE NEWS ----------------
def fetch_live_news():
    try:
        url = "https://newsapi.org/v2/top-headlines?country=us&apiKey=55f1615cfddb4e78b5856d5eb9f0f6b1"
        res = requests.get(url).json()
        return [a["title"] for a in res.get("articles", [])]
    except:
        return []

# ---------------- EXPLAIN ----------------
def explain_prediction(text, label):
    if label == "Fake":
        return "⚠ The model detected misleading or exaggerated patterns."
    else:
        return "✅ The text appears factual and structured."

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🧠 AI Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">BERT + FastAPI + Live News Integration</div>', unsafe_allow_html=True)

# ---------------- API STATUS UI ----------------
if check_api():
    st.success("🟢 API Connected")
else:
    st.error("🔴 API Not Connected")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyzer", "📜 History", "📊 Analytics", "🌐 Live News"])

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- ANALYZER ----------------
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    user_input = st.text_area("Enter News Text", height=180)

    col1, col2, col3 = st.columns(3)

    analyze = col1.button("🚀 Analyze", use_container_width=True)
    clear = col2.button("🧹 Clear", use_container_width=True)
    sample = col3.button("📄 Sample", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if sample:
        user_input = "Stock market hits all-time high due to strong economy"

    if clear:
        st.session_state.clear()
        st.rerun()

    if analyze:
        if user_input.strip() == "":
            st.warning("⚠ Enter text")
        else:
            with st.spinner("Analyzing..."):
                label, conf = predict_news_api(user_input)

            if label is None:
                st.stop()

            # RESULT
            if label == "Fake":
                st.markdown('<div class="result-box">⚠ Fake News</div>', unsafe_allow_html=True)
                st.error("Likely misleading content")
            else:
                st.markdown('<div class="result-box">📰 Real News</div>', unsafe_allow_html=True)
                st.success("Content appears credible")

            # CONFIDENCE
            st.progress(conf)
            st.caption(f"Confidence: {conf*100:.2f}%")

            # EXPLANATION
            st.info(explain_prediction(user_input, label))

            # SAVE
            st.session_state.history.append({
                "text": user_input[:100],
                "result": label,
                "confidence": round(conf*100, 2)
            })

# ---------------- HISTORY ----------------
with tab2:
    st.markdown("### 📜 History")

    if not st.session_state.history:
        st.info("No history")
    else:
        for item in reversed(st.session_state.history):
            st.write(item)

# ---------------- ANALYTICS ----------------
with tab3:
    st.markdown("### 📊 Analytics")

    df = pd.DataFrame(st.session_state.history)

    if not df.empty:
        st.bar_chart(df["confidence"])
        st.dataframe(df)
    else:
        st.info("No data yet")

# ---------------- LIVE NEWS ----------------
with tab4:
    st.markdown("### 🌐 Live News")

    news_list = fetch_live_news()

    for news in news_list[:5]:
        if st.button(news):
            label, conf = predict_news_api(news)

            if label is None:
                st.stop()

            if label == "Fake":
                st.error(f"{news} → Fake ⚠ ({conf})")
            else:
                st.success(f"{news} → Real 📰 ({conf})")