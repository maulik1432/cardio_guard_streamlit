import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")


@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        st.error("model.pkl was not found. Run train_model.py first and place model.pkl beside app.py.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


bundle = load_bundle()
model = bundle["model"]
metrics = load_metrics()


def build_features(age, gender, height, weight, ap_hi, ap_lo,
                    cholesterol, gluc, smoke, alco, active):
    height_m = height / 100.0
    bmi = weight / (height_m ** 2)
    pulse_pressure = ap_hi - ap_lo
    bp_ratio = ap_hi / ap_lo if ap_lo != 0 else np.nan

    return pd.DataFrame([{
        "age_years": float(age),
        "gender": int(gender),
        "height": float(height),
        "weight": float(weight),
        "ap_hi": float(ap_hi),
        "ap_lo": float(ap_lo),
        "cholesterol": int(cholesterol),
        "gluc": int(gluc),
        "smoke": int(smoke),
        "alco": int(alco),
        "active": int(active),
        "bmi": float(bmi),
        "pulse_pressure": float(pulse_pressure),
        "bp_ratio": float(bp_ratio),
    }])[bundle["feature_cols"]]


st.markdown("""
<style>
.stApp { background: #101819; color: #e7efed; color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"] { background-color: #101819; }
[data-testid="stHeader"] { background: rgba(16, 24, 25, .92); }
.block-container { max-width: 1240px; padding-top: 2.25rem; padding-bottom: 3rem; }
.hero { padding: 30px 34px; border-radius: 8px; background: #1b3030; color: #f5f8f7; margin-bottom: 22px; border: 1px solid #2c4946; border-left: 7px solid #e69a63; }
.hero h1 { color: #fff !important; font-size: 36px; margin: 0; font-weight: 750; }
.hero p { color: #b8ceca; margin: 7px 0 0; font-size: 16px; }
.eyebrow { color: #f0b47e; font-size: 12px; font-weight: 700; text-transform: uppercase; margin: 0 0 8px; }
.section-title { font-size: 20px; font-weight: 700; color: #8dd3c2; margin: 2px 0 4px; }
.section-copy { color: #a3b7b2; font-size: 14px; margin: 0 0 18px; }
.result-high { background: #382522; color: #ffe5dc; border: 1px solid #714638; border-left: 5px solid #f07858; padding: 20px 22px; border-radius: 6px; }
.result-low { background: #1d332d; color: #dcf2e8; border: 1px solid #3b6757; border-left: 5px solid #65c39c; padding: 20px 22px; border-radius: 6px; }
.result-high h2, .result-low h2 { margin: 0 0 6px; font-size: 22px; }
.result-high p, .result-low p { margin: 0; }
.footer { text-align: center; color: #80938f; padding-top: 28px; font-size: 12px; }
[data-testid="stMetric"] { background: #1b282a; border: 1px solid #304345; border-radius: 6px; padding: 14px 16px; }
[data-testid="stMetricLabel"] { color: #a4b9b4; }
[data-testid="stMetricValue"] { color: #f0f5f3; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #304345; border-radius: 8px; background: #172224; }
.stMarkdown, .stCaption, label, [data-testid="stWidgetLabel"] p { color: #dce7e4; }
[data-baseweb="input"] > div, [data-baseweb="select"] > div { background: #202e30; border-color: #405456; }
[data-baseweb="input"] input, [data-baseweb="select"] input { color: #f0f5f3; }
[data-baseweb="select"] svg { fill: #a8c0ba; }
[data-baseweb="popover"] [role="listbox"] { background: #202e30; }
[data-baseweb="popover"] [role="option"] { color: #e7efed; }
[data-baseweb="popover"] [role="option"]:hover { background: #304443; }
button[kind="primary"] { background: #dd7654; border-color: #dd7654; color: #171b1b; font-weight: 700; }
button[kind="primary"]:hover { background: #f08b67; border-color: #f08b67; color: #171b1b; }
[data-testid="stAlert"] { background: #332c20; border-color: #a78345; color: #f6e7c8; }
[data-testid="stProgressBar"] > div > div { background: #63bea2; }
a { color: #8dd3c2; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Cardiovascular health · Decision support</div>
  <h1>CardioGuard</h1>
  <p>A machine-learning estimate based on health and lifestyle information.</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Classifier", "HistGradientBoosting")
with m2:
    st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0)*100:.2f}%" if metrics else "—")
with m3:
    st.metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0):.3f}" if metrics else "—")
with m4:
    st.metric("Training records", f"{metrics.get('training_rows', 0):,}" if metrics else "—")

st.write("")
left, right = st.columns([1.55, 0.85], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Patient profile</div><div class="section-copy">Enter the available measurements and health indicators.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 18, 100, 50, 1)
        with c2:
            gender_label = st.selectbox("Gender", ["Female", "Male"])
            gender = 1 if gender_label == "Female" else 2
        with c3:
            height = st.number_input("Height (cm)", 100, 230, 170, 1)

        c4, c5, c6 = st.columns(3)
        with c4:
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.5)
        with c5:
            ap_hi = st.number_input("Systolic BP (mmHg)", 70, 240, 120, 1)
        with c6:
            ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 160, 80, 1)

        st.markdown('<div class="section-title">Clinical markers</div>', unsafe_allow_html=True)
        c7, c8 = st.columns(2)
        with c7:
            cholesterol_label = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
            cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol_label]
        with c8:
            gluc_label = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
            gluc = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc_label]

        st.markdown('<div class="section-title">Lifestyle</div>', unsafe_allow_html=True)
        c9, c10, c11 = st.columns(3)
        with c9:
            smoke_label = st.selectbox("Smoking", ["Non-Smoker", "Smoker"])
            smoke = 1 if smoke_label == "Smoker" else 0
        with c10:
            alco_label = st.selectbox("Alcohol use", ["No", "Yes"])
            alco = 1 if alco_label == "Yes" else 0
        with c11:
            active_label = st.selectbox("Physical activity", ["Active", "Inactive"])
            active = 1 if active_label == "Active" else 0

        predict = st.button("Assess risk", type="primary", use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown('<div class="section-title">About the model</div>', unsafe_allow_html=True)
        st.markdown(
            "**HistGradientBoosting** combines many small decision trees, each correcting errors made by earlier trees. "
            "It can learn non-linear patterns in tabular health data without feature scaling."
        )
        st.markdown("**How it was evaluated**")
        st.write(
            f"The classifier was tuned with 3-fold cross-validation on the training portion. "
            f"A held-out 20% test set ({metrics.get('test_rows', 0):,} records) produced "
            f"{metrics.get('test_accuracy', 0)*100:.1f}% accuracy and an ROC-AUC of "
            f"{metrics.get('test_roc_auc', 0):.3f}."
            if metrics else "The saved evaluation metrics are not available."
        )
        st.markdown("**What goes into a prediction**")
        st.caption("Age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol and activity. BMI, pulse pressure and blood-pressure ratio are derived from those values.")
        st.markdown("**Data handling**")
        st.caption("Missing numeric values are median-imputed. The training pipeline does not apply IQR outlier removal or calculate a noise score.")
        st.warning("For education only. This estimate is not a diagnosis or a substitute for advice from a qualified healthcare professional.")

if predict:
    if ap_lo >= ap_hi:
        st.error("Please enter a diastolic blood pressure lower than the systolic blood pressure.")
        st.stop()

    features = build_features(
        age, gender, height, weight, ap_hi, ap_lo,
        cholesterol, gluc, smoke, alco, active
    )

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])
    pct = probability * 100

    st.write("")
    st.markdown("## Assessment result")
    r1, r2 = st.columns([1, 1.2], gap="large")

    with r1:
        if prediction == 1:
            st.markdown('<div class="result-high"><h2>Elevated model estimate</h2><p>The classifier assigned this input to class 1.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-low"><h2>Lower model estimate</h2><p>The classifier assigned this input to class 0.</p></div>', unsafe_allow_html=True)
        st.metric("Model probability for class 1", f"{pct:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))

    with r2:
        height_m = height / 100.0
        bmi = weight / (height_m ** 2)
        pulse_pressure = ap_hi - ap_lo
        a, b, c = st.columns(3)
        with a:
            st.metric("BMI", f"{bmi:.1f}")
        with b:
            st.metric("Pulse pressure", f"{pulse_pressure} mmHg")
        with c:
            st.metric("Age", f"{age} yr")
        st.caption("The probability shown is the classifier's estimated probability, not a clinical probability of disease.")

st.markdown('<div class="footer">CardioGuard · HistGradientBoosting · Educational machine-learning project</div>', unsafe_allow_html=True)
