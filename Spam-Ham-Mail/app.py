import streamlit as st
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import pickle
import re
from nltk.corpus import stopwords
import nltk
from gtts import gTTS
import os
# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")

# --------------------------
# NLTK Stopwords
# --------------------------
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
EMOJI_PATTERN = re.compile("[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)

# --------------------------
# Load model
# --------------------------
@st.cache_resource
def load_model(path="spam_classifier.pkl"):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}")
        st.stop()  # Stop the app if model is missing
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved['model'], saved['vectorizer'], saved['num_cols']

clf, vectorizer, num_cols = load_model()

# --------------------------
# CSS Styling
# --------------------------
st.markdown("""
<style>
body { background-color: #f9fbfd; }
.gradient-header { background: linear-gradient(90deg, #36d1dc, #5b86e5); padding:25px; border-radius:15px; text-align:center; }
.gradient-header h1 { color:white; font-size:36px; margin:0; }
.gradient-header p { color:#f0f0f0; font-size:18px; }
.stButton>button { background: linear-gradient(90deg,#36d1dc,#5b86e5); color:white; font-weight:bold; border-radius:12px; padding:12px 30px; box-shadow:0 4px 6px rgba(0,0,0,0.2); transition:all 0.3s; }
.stButton>button:hover { background: linear-gradient(90deg,#5b86e5,#36d1dc); transform:scale(1.05); }
.result-card, .about-card, .sop-card { animation: fadeIn 1.5s; padding:20px; border-radius:10px; color:#111; }
.result-card h3, .result-card p, .about-card h4, .about-card p, .sop-card pre { color:#111; margin:5px 0; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Header
# --------------------------
st.markdown("""
<div class='gradient-header'>
    <h1>📧 AI-Powered Spam Email Classifier</h1>
    <p>Detect spam emails quickly & reliably</p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Predictor", "ℹ️ About", "📜 SOP"])

# --------------------------
# Helper Functions
# --------------------------
def count_emojis(s): return len(EMOJI_PATTERN.findall(str(s)))
def count_digits(s): return sum(c.isdigit() for c in str(s))
def uppercase_ratio(s):
    letters = [c for c in str(s) if c.isalpha()]
    return sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
def tokenize_simple(s): return re.findall(r"\w+", str(s).lower())
def add_features(df):
    df = df.copy()
    df['char_count'] = df['message'].str.len()
    df['word_count'] = df['message'].apply(lambda s: len(tokenize_simple(s)))
    df['digit_count'] = df['message'].apply(count_digits)
    df['digit_ratio'] = df['digit_count'] / df['char_count'].replace(0,1)
    df['emoji_count'] = df['message'].apply(count_emojis)
    df['upper_ratio'] = df['message'].apply(uppercase_ratio)
    df['punct_count'] = df['message'].apply(lambda s: sum(1 for c in str(s) if c in '!?.,:;'))
    df['stopword_count'] = df['message'].apply(lambda s: sum(1 for w in tokenize_simple(s) if w in STOPWORDS))
    return df

def predict_single(email, threshold=0.2):
    df = pd.DataFrame({'message':[email]})
    df_feat = add_features(df)
    X_text = vectorizer.transform(df_feat['message'])
    X_num = csr_matrix(df_feat[num_cols].values)
    X = hstack([X_text, X_num])
    prob = clf.predict_proba(X)[0,1]
    label = "spam" if prob >= threshold else "ham"
    return label, prob

def predict_batch(emails, threshold=0.2):
    df = pd.DataFrame({'message': emails})
    df_feat = add_features(df)
    X_text = vectorizer.transform(df_feat['message'])
    X_num = csr_matrix(df_feat[num_cols].values)
    X = hstack([X_text, X_num])
    probs = clf.predict_proba(X)[:,1]
    preds = ["spam" if p >= threshold else "ham" for p in probs]
    return pd.DataFrame({"Email": emails, "Predicted Label": preds, "Probability": probs.round(3)})

# --------------------------
# Tab 1: Predictor
# --------------------------
with tab1:
    st.sidebar.markdown("""
    <div style='background:#f0f2f6; color:#111; padding:15px; border-radius:10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);'>
        <h3>📘 Instructions</h3>
        <ul style="font-size:14px; margin:0; padding-left:18px;">
            <li>Enter a single email or multiple emails.</li>
            <li>Click <b>Predict</b> to check spam probability.</li>
            <li>Batch emails should be separated by semicolons `;`.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✍️ Enter Emails")
    single_email = st.text_area("Single Email:", height=120)
    batch_text = st.text_area("Multiple Emails (separate with `;`):", height=150)

    if st.button("🔍 Predict Emails", use_container_width=True):
        # ---- Single Email ----
        if single_email.strip():
            label, prob = predict_single(single_email)
            st.markdown(f"""
            <div class="result-card" style='background:#D5F5E3;border-left:6px solid #33cc33;'>
                <h3>Single Email</h3>
                <p>Predicted Label: <b>{label}</b></p>
                <p>Probability: <b>{prob:.3f}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # ---- Batch Emails ----
        if batch_text.strip():
            batch_emails = [e.strip() for e in batch_text.split(";") if e.strip()]
            if batch_emails:
                results = predict_batch(batch_emails)
                st.markdown("<h4>Batch Predictions</h4>", unsafe_allow_html=True)
                st.dataframe(results)
            else:
                st.warning("No valid emails found in batch input.")

# --------------------------
# Tab 2: About
# --------------------------
with tab2:
    st.header("ℹ️ About")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="about-card" style='background:white;padding:20px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
            <h4>📂 Dataset Info</h4>
            <p>Spam/Ham Email Dataset</p>
            <p>Features: Message text + engineered features</p>
            <p>Target: spam / ham</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="about-card" style='background:white;padding:20px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
            <h4>🛠 Tech Stack</h4>
            <p>🐍 Python</p>
            <p>🖥 Streamlit</p>
            <p>🧠 Logistic Regression / SVM / Custom Models</p>
            <p>📊 Scikit-learn, Pandas, NLTK</p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------
# Tab 3: SOP
# --------------------------
with tab3:
    st.header("📜 Standard Operating Procedure (SOP)")
    st.info("This tool is for **educational purposes only**.")

    sop_points = [
        "Input a single email or batch emails.",
        "Compute engineered features: char count, digits, emojis, etc.",
        "Predict probability of spam using ML model.",
        "Display predicted label and probability."
    ]

    sop_html = "<div class='sop-card' style='padding:15px;background:#fafafa;border-radius:10px;border:1px solid #ddd;'>"
    sop_html += "<ul style='color:#111;font-size:14px;'>"
    for point in sop_points:
        sop_html += f"<li>{point}</li>"
    sop_html += "</ul></div>"

    st.markdown(sop_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔊 Speak SOP", use_container_width=True):
            with st.spinner("Generating audio..."):
                try:
                    tts_text = "\n".join(sop_points)
                    tts = gTTS(text=tts_text, lang='en')
                    audio_path = "sop_tts.mp3"
                    tts.save(audio_path)
                    st.success("✅ SOP Audio Generated")
                    st.audio(audio_path, format='audio/mp3')
                except Exception as e:
                    st.error("⚠️ Failed to connect to Google TTS. Check internet.")
                    st.caption(f"Error: {e}")

# --------------------------
# Footer
# --------------------------
st.markdown("""
<div style='margin-top:30px;padding:10px;
            background:linear-gradient(to right,#36d1dc,#5b86e5);
            border-radius:10px;text-align:center;color:white;font-size:13px;'>
    Made with ❤️ using Streamlit & Python | For Educational Purposes Only
</div>
""", unsafe_allow_html=True)


