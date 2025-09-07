import streamlit as st
import pandas as pd
import numpy as np
import pickle
from gtts import gTTS

# ----------------------
# Load pre-trained models
# ----------------------
with open("logreg_diabetes_model.pkl", "rb") as f:
    logreg_saved = pickle.load(f)
logreg_model = logreg_saved["model"]
logreg_threshold = logreg_saved["threshold"]

with open("rf_diabetes_model.pkl", "rb") as f:
    rf_saved = pickle.load(f)
rf_model = rf_saved["model"]
rf_threshold = rf_saved["threshold"]

# ----------------------
# Streamlit Config
# ----------------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# ----------------------
# Custom CSS
# ----------------------
st.markdown("""
<style>
body { background-color: #f9fbfd; }
.gradient-header { background: linear-gradient(90deg, #36d1dc, #5b86e5); padding:25px; border-radius:15px; text-align:center; }
.gradient-header h1 { color:white; font-size:36px; margin:0; }
.gradient-header p { color:#f0f0f0; font-size:18px; }
.stButton>button { background: linear-gradient(90deg,#36d1dc,#5b86e5); color:white; font-weight:bold; border-radius:12px; padding:12px 30px; box-shadow:0 4px 6px rgba(0,0,0,0.2); transition:all 0.3s; }
.stButton>button:hover { background: linear-gradient(90deg,#5b86e5,#36d1dc); transform:scale(1.05); }

.result-card, .about-card, .sop-card { 
    animation: fadeIn 1.5s; 
    padding:20px; 
    border-radius:10px; 
    color:#111;  /* Force text to dark */
}
.result-card h3, .result-card p, 
.about-card h4, .about-card p,
.sop-card pre {
    color:#111; 
    margin:5px 0; 
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Header
# ----------------------
st.markdown("""
<div class='gradient-header'>
    <h1>🧬 AI-Powered Diabetes Prediction</h1>
    <p>Smart, Fast & Reliable Health Insights</p>
</div>
""", unsafe_allow_html=True)

# ----------------------
# Tabs
# ----------------------
tab1, tab2, tab3 = st.tabs(["🔍 Predictor", "ℹ️ About", "📜 SOP"])

# ----------------------
# Helper Functions
# ----------------------
def get_risk_level(prob):
    if prob < 0.3: return "Low"
    elif prob < 0.6: return "Medium"
    else: return "High"

def get_color(risk):
    return {"Low":"🟢","Medium":"🟡","High":"🔴"}[risk]

# ----------------------
# Tab 1: Predictor
# ----------------------
with tab1:
    st.sidebar.markdown("""
    <div style='background:#f0f2f6; color:#111; padding:15px; border-radius:10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);'>
        <h3>📘 Instructions</h3>
        <ul style="font-size:14px; margin:0; padding-left:18px;">
            <li>Adjust values using sliders.</li>
            <li>Click <b>Predict</b> to check diabetes risk.</li>
            <li>Results show probabilities & risk levels.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)


    st.markdown("### ✍️ Enter Your Health Data")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.slider("🤰 Pregnancies", 0, 20, 1)
            glucose = st.slider("🩸 Glucose Level", 50, 200, 120)
            bp = st.slider("💓 Blood Pressure", 40, 140, 70)
            skin = st.slider("🧪 Skin Thickness (optional)", 0, 100, 20)
        with col2:
            insulin = st.slider("💉 Insulin Level", 0, 900, 85)
            bmi = st.slider("⚖️ BMI", 10.0, 70.0, 25.0)
            dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)
            age = st.slider("🎂 Age", 10, 100, 30)

    if st.button("🔍 Predict", use_container_width=True):
        # ---- Derived Features ----
        bmi_age = bmi * age
        insulin_log = np.log1p(insulin) if insulin > 0 else 0.0

        # ---- Prepare DataFrame for prediction ----
        input_df = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
            "BMI_Age": bmi_age,
            "Insulin_log": insulin_log
        }])

        # ---- Logistic Regression Prediction ----
        prob_log = logreg_model.predict_proba(input_df)[:, 1][0]
        risk_log = get_risk_level(prob_log)
        color_log = get_color(risk_log)

        # ---- Random Forest Prediction ----
        prob_rf = rf_model.predict_proba(input_df)[:, 1][0]
        risk_rf = get_risk_level(prob_rf)
        color_rf = get_color(risk_rf)

        # ---- Display Results ----
        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="result-card" style='background:#D5F5E3;border-left:6px solid #33cc33;'>
                <h3>Logistic Regression</h3>
                <p>Probability: <b>{prob_log*100:.1f}% {color_log}</b></p>
                <p>Risk Level: <b>{risk_log}</b></p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="result-card" style='background:#D6EAF8;border-left:6px solid #3399ff;'>
                <h3>Random Forest</h3>
                <p>Probability: <b>{prob_rf*100:.1f}% {color_rf}</b></p>
                <p>Risk Level: <b>{risk_rf}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # ---- Recommendation ----
        st.markdown("""
        <div class="result-card" style='background:#FDEBD0;border-left:6px solid #ff9900;'>
            <b>Recommendation:</b> Logistic Regression is recommended because Random Forest may decrease accuracy after certain recall thresholds.
        </div>
        """, unsafe_allow_html=True)

# ----------------------
# Tab 2: About
# ----------------------
with tab2:
    st.header("ℹ️ About")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="about-card" style='background:white;padding:20px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
                <h4>📂 Dataset Info</h4>
                <p><b>Source:</b> Pima Indians Diabetes Dataset</p>
                <p><b>Features:</b> Glucose, Blood Pressure, BMI, Insulin, Age...</p>
                <p><b>Target:</b> Outcome (Diabetic / Not Diabetic)</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="about-card" style='background:white;padding:20px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
                <h4>🛠 Tech Stack</h4>
                <p>🐍 Python</p>
                <p>🖥 Streamlit</p>
                <p>🧠 Logistic Regression & Random Forest</p>
                <p>📊 Scikit-learn, Matplotlib</p>
            </div>
        """, unsafe_allow_html=True)

# ----------------------
# Tab 3: SOP
# ----------------------
with tab3:
    st.header("📜 Standard Operating Procedure (SOP)")
    st.info("This tool is for **educational purposes only** and should not replace professional medical advice.")

    sop_points = [
        "Input health data from user.",
        "Compute derived features: BMI_Age, Insulin_log.",
        "Predict with Logistic Regression & Random Forest.",
        "Show prediction probability, risk level, and recommendation."
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

# ----------------------
# Footer
# ----------------------
st.markdown("""
<div style='margin-top:30px;padding:10px;
            background:linear-gradient(to right,#36d1dc,#5b86e5);
            border-radius:10px;text-align:center;color:white;font-size:13px;'>
    Made with ❤️ using Streamlit & Python  | For Educational Purposes Only | Me used GPT to only on the topics I know , Like its my coding partner
</div>
""", unsafe_allow_html=True)
