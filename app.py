import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioPredict AI | Heart Disease Analytics & Oracle",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Clean, Modern Clinical Dashboard)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-container {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef2f9 100%);
        border: 1px solid #e1e7f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #059669;
        margin-top: 4px;
    }
    
    /* Risk Alert Badges */
    .risk-high-card {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 15px 0;
    }
    .risk-low-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 15px 0;
    }
    .oracle-card {
        background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%);
        border: 2px solid #c084fc;
        border-radius: 14px;
        padding: 24px;
        margin: 15px 0;
    }
    .badge-high {
        background-color: #e11d48;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        margin-bottom: 10px;
    }
    .badge-low {
        background-color: #16a34a;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        margin-bottom: 10px;
    }
    .badge-oracle {
        background: linear-gradient(90deg, #9333ea 0%, #c026d3 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 6px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Data and Serialized Models
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_resource
def load_models_and_metrics():
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))
    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.joblib"))
    svm = joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
    return {"Random Forest": rf, "Logistic Regression": lr, "Support Vector Machine": svm}, metrics

@st.cache_data
def load_dataset():
    csv_path = os.path.join(DATA_DIR, "heart.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

models_dict, metrics_data = load_models_and_metrics()
df_data = load_dataset()

# ---------------------------------------------------------
# Sidebar Configuration & Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/865/865969.png", width=70)
    st.title("CardioPredict AI")
    st.caption("Clinical Intelligence & Risk Stratification System")
    st.markdown("---")
    
    # Model Selection
    selected_model_name = st.selectbox(
        "⚡ Choose Machine Learning Model",
        options=["Random Forest", "Logistic Regression", "Support Vector Machine"],
        index=0,
        help="Select the trained algorithm used for inference."
    )
    
    current_acc = metrics_data[selected_model_name]['accuracy'] * 100
    st.success(f"**Selected Model:** {selected_model_name}\n\n**Test Accuracy:** `{current_acc:.1f}%`")
    
    st.markdown("---")
    # Navigation Radio
    navigation = st.radio(
        "Navigation",
        options=[
            "🩺 Individual Diagnosis",
            "🔮 Oracle Predictor",
            "📁 Batch Patient Screening",
            "📊 Model Benchmarks & XAI",
            "📈 Exploratory Data Analysis (EDA)",
            "📖 Clinical Dictionary & Info"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.caption(f"**Total Records:** {metrics_data['dataset_stats']['total_samples']}")
    st.caption(f"**Clinical Features:** {metrics_data['dataset_stats']['features_count']}")
    st.caption(f"**Disease Cases (1):** {metrics_data['dataset_stats']['target_distribution']['1']}")
    st.caption(f"**Healthy Cases (0):** {metrics_data['dataset_stats']['target_distribution']['0']}")
    
    st.markdown("---")
    st.caption("Powered by Scikit-Learn, Streamlit & Plotly")

# Active Model instance
active_model = models_dict[selected_model_name]

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); padding: 22px 28px; border-radius: 14px; color: white; margin-bottom: 25px;">
    <div>
        <h1 style="color: white; margin: 0; font-size: 2.1rem; font-weight: 800;">CardioPredict: Heart Disease Risk Assessment</h1>
        <p style="color: #bfdbfe; margin: 6px 0 0 0; font-size: 1.05rem;">
            Real-time interactive diagnostic analytics trained on clinical cardiovascular profiles.
        </p>
    </div>
    <div style="text-align: right;">
        <span style="background: rgba(255,255,255,0.2); padding: 8px 14px; border-radius: 8px; font-weight: 600; font-size: 0.9rem;">
            ML Engine: Online
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 1: Individual Diagnosis
# ---------------------------------------------------------
if navigation == "🩺 Individual Diagnosis":
    st.markdown('<div class="section-title">🩺 Single Patient Clinical Evaluation Form</div>', unsafe_allow_html=True)
    st.write("Adjust patient clinical vitals and test results below to evaluate the probability and risk level of heart disease.")
    
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics & Baseline")
            age = st.slider("Patient Age (years)", min_value=20, max_value=85, value=52, help="Age in completed years")
            sex_option = st.radio("Sex / Gender", options=["Male (1)", "Female (0)"], horizontal=True)
            sex = 1 if "Male" in sex_option else 0
            
            trestbps = st.slider(
                "Resting Blood Pressure (trestbps)",
                min_value=80, max_value=210, value=125,
                help="Resting blood pressure in mm Hg upon hospital admission. Normal: < 120 mm Hg"
            )
            chol = st.slider(
                "Serum Cholesterol (chol)",
                min_value=100, max_value=570, value=212,
                help="Serum cholesterol in mg/dl. Desirable: < 200 mg/dl"
            )
            fbs_option = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl (fbs)",
                options=["False (≤ 120 mg/dl)", "True (> 120 mg/dl)"],
                help="Indicator of elevated fasting blood sugar or diabetes risk."
            )
            fbs = 1 if "True" in fbs_option else 0

        with col2:
            st.markdown("#### 🫀 Symptoms & Stress Test")
            cp_map = {
                "0: Typical Angina": 0,
                "1: Atypical Angina": 1,
                "2: Non-Anginal Pain": 2,
                "3: Asymptomatic": 3
            }
            cp_label = st.selectbox("Chest Pain Type (cp)", options=list(cp_map.keys()), index=0)
            cp = cp_map[cp_label]
            
            thalach = st.slider(
                "Max Heart Rate Achieved (thalach)",
                min_value=60, max_value=220, value=168,
                help="Peak heart rate recorded during cardiac stress test."
            )
            exang_option = st.radio("Exercise Induced Angina (exang)", options=["No (0)", "Yes (1)"], horizontal=True)
            exang = 1 if "Yes" in exang_option else 0
            
            oldpeak = st.slider(
                "ST Depression Induced by Exercise (oldpeak)",
                min_value=0.0, max_value=6.5, value=1.0, step=0.1,
                help="ST depression induced by exercise relative to rest."
            )
            slope_map = {
                "0: Upsloping": 0,
                "1: Flat": 1,
                "2: Downsloping": 2
            }
            slope_label = st.selectbox("Slope of Peak Exercise ST (slope)", options=list(slope_map.keys()), index=2)
            slope = slope_map[slope_label]

        with col3:
            st.markdown("#### 🔬 Cardiac Diagnostics & Labs")
            restecg_map = {
                "0: Normal": 0,
                "1: ST-T Wave Abnormality": 1,
                "2: Left Ventricular Hypertrophy": 2
            }
            restecg_label = st.selectbox("Resting ECG Results (restecg)", options=list(restecg_map.keys()), index=1)
            restecg = restecg_map[restecg_label]
            
            ca = st.slider(
                "Major Vessels Colored by Fluoroscopy (ca)",
                min_value=0, max_value=4, value=2,
                help="Number of major vessels (0-4) visible under fluoroscopy."
            )
            
            thal_map = {
                "0: Normal / Null": 0,
                "1: Fixed Defect": 1,
                "2: Normal Flow": 2,
                "3: Reversible Defect": 3
            }
            thal_label = st.selectbox("Thalassemia Blood Disorder (thal)", options=list(thal_map.keys()), index=3)
            thal = thal_map[thal_label]
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔍 Run Heart Disease Risk Assessment", use_container_width=True, type="primary")

    if submitted:
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }])
        
        prediction = active_model.predict(input_data)[0]
        probabilities = active_model.predict_proba(input_data)[0]
        disease_prob = probabilities[1] * 100
        healthy_prob = probabilities[0] * 100
        
        st.markdown("---")
        st.markdown("### 📋 Clinical Risk Assessment Report")
        
        res_col1, res_col2 = st.columns([1.1, 1])
        
        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high-card">
                    <span class="badge-high">⚠️ HIGH RISK DETECTED</span>
                    <h2 style="color: #9f1239; margin: 10px 0 5px 0;">Heart Disease Indicated</h2>
                    <p style="color: #475569; font-size: 1.05rem; margin-bottom: 12px;">
                        The model predicts a <strong>{disease_prob:.1f}% probability</strong> of coronary artery disease presence.
                    </p>
                    <div style="text-align: left; background: white; padding: 14px 18px; border-radius: 8px; border-left: 4px solid #e11d48; margin-top: 15px;">
                        <strong style="color: #9f1239;">Suggested Clinical Action:</strong>
                        <ul style="margin: 6px 0 0 18px; color: #334155; font-size: 0.9rem;">
                            <li>Prioritize patient for non-invasive imaging (Echocardiography or CT Angiogram).</li>
                            <li>Re-evaluate cardiovascular risk factors and lipid panel.</li>
                            <li>Consult cardiology for formal diagnostic protocol.</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low-card">
                    <span class="badge-low">✅ LOW RISK</span>
                    <h2 style="color: #14532d; margin: 10px 0 5px 0;">No Heart Disease Detected</h2>
                    <p style="color: #475569; font-size: 1.05rem; margin-bottom: 12px;">
                        The model predicts a <strong>{healthy_prob:.1f}% probability</strong> of healthy cardiac status.
                    </p>
                    <div style="text-align: left; background: white; padding: 14px 18px; border-radius: 8px; border-left: 4px solid #16a34a; margin-top: 15px;">
                        <strong style="color: #14532d;">Clinical Recommendation:</strong>
                        <ul style="margin: 6px 0 0 18px; color: #334155; font-size: 0.9rem;">
                            <li>Maintain standard annual health checkups and preventative screenings.</li>
                            <li>Continue regular aerobic exercise and balanced dietary habits.</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            # Interactive Gauge Meter
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=disease_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Risk Probability Gauge ({selected_model_name})", 'font': {'size': 18, 'color': '#0f172a'}},
                number={'suffix': "%", 'font': {'size': 36, 'color': '#0f172a'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                    'bar': {'color': "#f43f5e" if prediction == 1 else "#22c55e"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, 40], 'color': '#dcfce7'},
                        {'range': [40, 70], 'color': '#fef3c7'},
                        {'range': [70, 100], 'color': '#fee2e2'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Contributing Vitals Table
        st.markdown("#### 🔍 Clinical Factor Breakdown")
        factor_col1, factor_col2, factor_col3, factor_col4 = st.columns(4)
        factor_col1.metric("Resting BP", f"{trestbps} mm Hg", delta="Elevated" if trestbps > 130 else "Normal", delta_color="inverse")
        factor_col2.metric("Cholesterol", f"{chol} mg/dl", delta="High" if chol > 240 else "Normal", delta_color="inverse")
        factor_col3.metric("Max Heart Rate", f"{thalach} bpm", delta="Normal" if thalach >= 140 else "Sub-optimal")
        factor_col4.metric("ST Depression", f"{oldpeak}", delta="High Risk" if oldpeak > 1.5 else "Low Risk", delta_color="inverse")

# ---------------------------------------------------------
# TAB 2: Oracle Predictor (NEW FEATURE)
# ---------------------------------------------------------
elif navigation == "🔮 Oracle Predictor":
    st.markdown('<div class="section-title">🔮 The Cardio-Oracle: Multi-Model Consensus & Counterfactual Simulator</div>', unsafe_allow_html=True)
    st.write("The **Oracle Predictor** queries all 3 machine learning algorithms simultaneously, builds a weighted consensus verdict, and runs real-time **'What-If' clinical counterfactual simulations** to determine the exact lifestyle and medical interventions needed to reverse cardiovascular risk.")
    
    # Preset Patient Profiles for quick experimentation
    st.markdown("#### ⚡ Quick Patient Preset Selector")
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    
    preset = "Custom"
    if preset_col1.button("🚨 High-Risk Hypertensive Patient", use_container_width=True):
        st.session_state['o_age'] = 62
        st.session_state['o_sex'] = 1
        st.session_state['o_cp'] = 0
        st.session_state['o_trestbps'] = 160
        st.session_state['o_chol'] = 286
        st.session_state['o_fbs'] = 1
        st.session_state['o_restecg'] = 2
        st.session_state['o_thalach'] = 108
        st.session_state['o_exang'] = 1
        st.session_state['o_oldpeak'] = 2.8
        st.session_state['o_slope'] = 1
        st.session_state['o_ca'] = 3
        st.session_state['o_thal'] = 3
    elif preset_col2.button("⚠️ Borderline Cardiac Patient", use_container_width=True):
        st.session_state['o_age'] = 54
        st.session_state['o_sex'] = 1
        st.session_state['o_cp'] = 1
        st.session_state['o_trestbps'] = 135
        st.session_state['o_chol'] = 240
        st.session_state['o_fbs'] = 0
        st.session_state['o_restecg'] = 1
        st.session_state['o_thalach'] = 145
        st.session_state['o_exang'] = 0
        st.session_state['o_oldpeak'] = 1.2
        st.session_state['o_slope'] = 1
        st.session_state['o_ca'] = 1
        st.session_state['o_thal'] = 2
    elif preset_col3.button("🟢 Healthy Athletic Individual", use_container_width=True):
        st.session_state['o_age'] = 38
        st.session_state['o_sex'] = 0
        st.session_state['o_cp'] = 2
        st.session_state['o_trestbps'] = 112
        st.session_state['o_chol'] = 175
        st.session_state['o_fbs'] = 0
        st.session_state['o_restecg'] = 0
        st.session_state['o_thalach'] = 182
        st.session_state['o_exang'] = 0
        st.session_state['o_oldpeak'] = 0.0
        st.session_state['o_slope'] = 2
        st.session_state['o_ca'] = 0
        st.session_state['o_thal'] = 2
    elif preset_col4.button("🔄 Reset to Default Baseline", use_container_width=True):
        for k in ['o_age', 'o_sex', 'o_cp', 'o_trestbps', 'o_chol', 'o_fbs', 'o_restecg', 'o_thalach', 'o_exang', 'o_oldpeak', 'o_slope', 'o_ca', 'o_thal']:
            if k in st.session_state:
                del st.session_state[k]

    # Baseline Inputs
    with st.expander("🛠️ View / Adjust Baseline Patient Vitals", expanded=True):
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            o_age = st.slider("Age (years)", 20, 85, st.session_state.get('o_age', 56), key="s_age")
            o_sex_label = st.radio("Sex", ["Male (1)", "Female (0)"], index=0 if st.session_state.get('o_sex', 1) == 1 else 1, horizontal=True)
            o_sex = 1 if "Male" in o_sex_label else 0
            o_trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 210, st.session_state.get('o_trestbps', 140), key="s_bp")
            o_chol = st.slider("Serum Cholesterol (mg/dl)", 100, 570, st.session_state.get('o_chol', 250), key="s_chol")
        with bcol2:
            cp_opts = ["0: Typical Angina", "1: Atypical Angina", "2: Non-Anginal Pain", "3: Asymptomatic"]
            o_cp_idx = st.session_state.get('o_cp', 0)
            o_cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: cp_opts[x], index=o_cp_idx, key="s_cp")
            o_thalach = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, st.session_state.get('o_thalach', 130), key="s_hr")
            o_exang_idx = st.session_state.get('o_exang', 1)
            o_exang = st.radio("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", index=o_exang_idx, horizontal=True, key="s_exang")
            o_oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, float(st.session_state.get('o_oldpeak', 1.8)), step=0.1, key="s_op")
        with bcol3:
            slope_opts = ["0: Upsloping", "1: Flat", "2: Downsloping"]
            o_slope_idx = st.session_state.get('o_slope', 1)
            o_slope = st.selectbox("ST Slope", options=[0, 1, 2], format_func=lambda x: slope_opts[x], index=o_slope_idx, key="s_slope")
            o_ca = st.slider("Major Fluoroscopy Vessels (0-4)", 0, 4, st.session_state.get('o_ca', 2), key="s_ca")
            thal_opts = ["0: Normal/Null", "1: Fixed Defect", "2: Normal Flow", "3: Reversible Defect"]
            o_thal_idx = st.session_state.get('o_thal', 3)
            o_thal = st.selectbox("Thalassemia Scan", options=[0, 1, 2, 3], format_func=lambda x: thal_opts[x], index=o_thal_idx, key="s_thal")
            o_fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: "True (> 120)" if x == 1 else "False (≤ 120)", index=st.session_state.get('o_fbs', 0), key="s_fbs")
            o_restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: ["0: Normal", "1: ST-T Abnormality", "2: LV Hypertrophy"][x], index=st.session_state.get('o_restecg', 1), key="s_ecg")

    # Baseline DataFrame
    baseline_df = pd.DataFrame([{
        'age': o_age, 'sex': o_sex, 'cp': o_cp, 'trestbps': o_trestbps, 'chol': o_chol,
        'fbs': o_fbs, 'restecg': o_restecg, 'thalach': o_thalach, 'exang': o_exang,
        'oldpeak': o_oldpeak, 'slope': o_slope, 'ca': o_ca, 'thal': o_thal
    }])

    # 1. Oracle Multi-Model Consensus Engine
    rf_prob = models_dict["Random Forest"].predict_proba(baseline_df)[0][1] * 100
    lr_prob = models_dict["Logistic Regression"].predict_proba(baseline_df)[0][1] * 100
    svm_prob = models_dict["Support Vector Machine"].predict_proba(baseline_df)[0][1] * 100

    rf_pred = models_dict["Random Forest"].predict(baseline_df)[0]
    lr_pred = models_dict["Logistic Regression"].predict(baseline_df)[0]
    svm_pred = models_dict["Support Vector Machine"].predict(baseline_df)[0]

    # Weighted Ensemble Oracle Calculation (RF weight: 50%, LR weight: 25%, SVM weight: 25%)
    oracle_risk_score = (0.50 * rf_prob) + (0.25 * lr_prob) + (0.25 * svm_prob)
    votes_disease = int(rf_pred + lr_pred + svm_pred)

    st.markdown("---")
    st.markdown("### 🏛️ Part 1: Oracle Multi-Model Consensus Verdict")

    o_col_left, o_col_right = st.columns([1.2, 1])

    with o_col_left:
        if votes_disease == 3:
            verdict_badge = '<span class="badge-high">🚨 UNANIMOUS HIGH RISK (3/3 MODELS AGREE)</span>'
            verdict_text = "All three independent classification algorithms classify this patient as <strong>High Risk for Coronary Heart Disease</strong>."
            card_class = "risk-high-card"
        elif votes_disease == 2:
            verdict_badge = '<span class="badge-high" style="background-color: #d97706;">⚠️ MAJORITY HIGH RISK (2/3 MODELS AGREE)</span>'
            verdict_text = "Two out of three models indicate <strong>elevated heart disease risk</strong>. Further clinical investigation is warranted."
            card_class = "risk-high-card"
        elif votes_disease == 1:
            verdict_badge = '<span class="badge-low" style="background-color: #0284c7;">🟡 BORDERLINE / MAJORITY LOW RISK (2/3 MODELS HEALTHY)</span>'
            verdict_text = "Two out of three models predict <strong>low risk</strong>, but one model flagged borderline anomalies."
            card_class = "risk-low-card"
        else:
            verdict_badge = '<span class="badge-low">✅ UNANIMOUS LOW RISK (3/3 MODELS AGREE)</span>'
            verdict_text = "All three algorithms independently agree: <strong>no evidence of significant heart disease</strong>."
            card_class = "risk-low-card"

        st.markdown(f"""
        <div class="{card_class}">
            {verdict_badge}
            <h2 style="margin: 8px 0; color: #0f172a;">Oracle Consensus Score: {oracle_risk_score:.1f}%</h2>
            <p style="color: #334155; font-size: 1.05rem;">{verdict_text}</p>
        </div>
        """, unsafe_allow_html=True)

        # Model individual scores
        m1, m2, m3 = st.columns(3)
        m1.metric("🌲 Random Forest", f"{rf_prob:.1f}%", delta="High Risk" if rf_pred == 1 else "Low Risk", delta_color="inverse")
        m2.metric("📈 Logistic Reg.", f"{lr_prob:.1f}%", delta="High Risk" if lr_pred == 1 else "Low Risk", delta_color="inverse")
        m3.metric("⚡ Support Vector", f"{svm_prob:.1f}%", delta="High Risk" if svm_pred == 1 else "Low Risk", delta_color="inverse")

    with o_col_right:
        fig_oracle_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=oracle_risk_score,
            title={'text': "Oracle Consensus Risk Index", 'font': {'size': 18, 'color': '#6b21a8'}},
            number={'suffix': "%", 'font': {'size': 38, 'color': '#4a044e'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#9333ea"},
                'steps': [
                    {'range': [0, 35], 'color': '#dcfce7'},
                    {'range': [35, 65], 'color': '#fef3c7'},
                    {'range': [65, 100], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': "purple", 'width': 4},
                    'thickness': 0.8,
                    'value': 50
                }
            }
        ))
        fig_oracle_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_oracle_gauge, use_container_width=True)

    # 2. Interactive "What-If" Counterfactual Intervention Engine
    st.markdown("---")
    st.markdown("### 🧬 Part 2: Oracle 'What-If' Counterfactual Simulator")
    st.write("Simulate how targeted clinical treatments and lifestyle changes will alter the patient's future cardiac risk trajectory:")

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        st.markdown("#### 🎯 Clinical Intervention Controls")
        sim_bp_reduction = st.slider("Blood Pressure Reduction (mm Hg)", 0, 60, 20 if o_trestbps > 130 else 0, help="Antihypertensive therapy or sodium restriction")
        sim_chol_reduction = st.slider("Cholesterol Lowering (mg/dl)", 0, 120, 50 if o_chol > 220 else 0, help="Statin therapy or dietary adjustment")
        sim_hr_improvement = st.slider("Cardio Fitness / Max HR Boost (bpm)", 0, 40, 20 if o_thalach < 150 else 0, help="Aerobic cardiac rehab")
        sim_op_reduction = st.slider("ST Depression Improvement (units)", 0.0, 3.0, min(1.0, o_oldpeak), step=0.1, help="Revascularization or ischemia relief")
        sim_stop_angina = st.checkbox("Relieve Exercise Angina (exang = 0)", value=(o_exang == 1))

        # Compute simulated patient values
        sim_trestbps = max(90, o_trestbps - sim_bp_reduction)
        sim_chol = max(120, o_chol - sim_chol_reduction)
        sim_thalach = min(210, o_thalach + sim_hr_improvement)
        sim_oldpeak = max(0.0, o_oldpeak - sim_op_reduction)
        sim_exang = 0 if sim_stop_angina else o_exang

        simulated_df = baseline_df.copy()
        simulated_df['trestbps'] = sim_trestbps
        simulated_df['chol'] = sim_chol
        simulated_df['thalach'] = sim_thalach
        simulated_df['oldpeak'] = sim_oldpeak
        simulated_df['exang'] = sim_exang

        # Predict post-intervention
        sim_rf_prob = models_dict["Random Forest"].predict_proba(simulated_df)[0][1] * 100
        sim_lr_prob = models_dict["Logistic Regression"].predict_proba(simulated_df)[0][1] * 100
        sim_svm_prob = models_dict["Support Vector Machine"].predict_proba(simulated_df)[0][1] * 100
        sim_oracle_score = (0.50 * sim_rf_prob) + (0.25 * sim_lr_prob) + (0.25 * sim_svm_prob)
        risk_reduction = oracle_risk_score - sim_oracle_score

    with sim_col2:
        st.markdown("#### 📊 Pre vs. Post-Intervention Trajectory")
        
        comp_chart_df = pd.DataFrame({
            "Scenario": ["Current Baseline", "Simulated Post-Intervention"],
            "Risk Score (%)": [oracle_risk_score, sim_oracle_score],
            "Risk Status": ["High Risk" if oracle_risk_score >= 50 else "Low Risk", "High Risk" if sim_oracle_score >= 50 else "Low Risk"]
        })

        fig_comp = px.bar(
            comp_chart_df,
            x="Scenario",
            y="Risk Score (%)",
            color="Risk Status",
            text="Risk Score (%)",
            color_discrete_map={"High Risk": "#f43f5e", "Low Risk": "#22c55e"},
            title="Oracle Projected Risk Trajectory"
        )
        fig_comp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_comp.update_layout(yaxis=dict(range=[0, 115]), height=320)
        st.plotly_chart(fig_comp, use_container_width=True)

        if risk_reduction > 0:
            st.success(f"🎉 **Positive Outcome Projected:** Interventions achieve a **-{risk_reduction:.1f}% net risk reduction** (from `{oracle_risk_score:.1f}%` down to `{sim_oracle_score:.1f}%`)!")
        else:
            st.info("No active interventions simulated. Adjust the sliders on the left to view predicted risk reduction.")

    # 3. Oracle Recommended Action Plan
    st.markdown("---")
    st.markdown("### 📋 Part 3: Oracle Clinical Action Plan")
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    rec_col1.markdown("""
    <div style="background: white; border-left: 4px solid #3b82f6; padding: 14px 18px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
        <strong style="color: #1d4ed8;">Priority 1: Ischemia Management</strong>
        <p style="font-size: 0.88rem; color: #475569; margin: 6px 0 0 0;">
            Targeting ST depression and resolving exercise angina yields the largest statistical drop in disease probability (~25-35% risk reduction).
        </p>
    </div>
    """, unsafe_allow_html=True)
    rec_col2.markdown("""
    <div style="background: white; border-left: 4px solid #10b981; padding: 14px 18px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
        <strong style="color: #047857;">Priority 2: Blood Pressure Control</strong>
        <p style="font-size: 0.88rem; color: #475569; margin: 6px 0 0 0;">
            Maintain systolic BP below 120 mm Hg. Each 10 mm Hg reduction decreases vascular wall strain and improves coronary perfusion.
        </p>
    </div>
    """, unsafe_allow_html=True)
    rec_col3.markdown("""
    <div style="background: white; border-left: 4px solid #8b5cf6; padding: 14px 18px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
        <strong style="color: #6d28d9;">Priority 3: Lipid Optimization</strong>
        <p style="font-size: 0.88rem; color: #475569; margin: 6px 0 0 0;">
            Target total serum cholesterol under 200 mg/dl and LDL under 70 mg/dl to arrest coronary atherosclerotic plaque progression.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: Batch Patient Screening
# ---------------------------------------------------------
elif navigation == "📁 Batch Patient Screening":
    st.markdown('<div class="section-title">📁 High-Throughput Batch Patient Screening</div>', unsafe_allow_html=True)
    st.write("Upload a `.csv` containing patient cardiovascular profiles to predict risk scores for multiple patients simultaneously.")
    
    col_upload, col_sample = st.columns([2.5, 1])
    
    with col_sample:
        st.markdown("#### 📥 Sample Template")
        st.write("Download this verified sample CSV with 20 test patient records to test the batch screening tool:")
        sample_path = os.path.join(DATA_DIR, "sample_batch_test.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Sample Batch CSV",
                    data=f,
                    file_name="sample_patients_batch.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    with col_upload:
        st.markdown("#### 📤 Upload Patient Batch Data")
        uploaded_file = st.file_uploader("Upload patient dataset (.csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded file with **{batch_df.shape[0]} rows** and **{batch_df.shape[1]} columns**.")
            
            # Verify feature requirements
            expected_cols = metrics_data['features']
            missing_cols = [c for c in expected_cols if c not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in uploaded CSV: `{missing_cols}`")
            else:
                features_data = batch_df[expected_cols]
                preds = active_model.predict(features_data)
                probs = active_model.predict_proba(features_data)[:, 1]
                
                results_df = batch_df.copy()
                results_df['Predicted_Risk'] = ["High Risk (Disease Detected)" if p == 1 else "Low Risk (Healthy)" for p in preds]
                results_df['Risk_Probability_%'] = (probs * 100).round(2)
                results_df['Triage_Priority'] = [
                    "🔴 Urgent" if pr > 0.75 else ("🟡 Moderate" if pr > 0.40 else "🟢 Normal") for pr in probs
                ]
                
                # KPIs
                high_count = (preds == 1).sum()
                low_count = (preds == 0).sum()
                avg_risk = probs.mean() * 100
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Total Screened</div>
                    <div class="metric-value">{len(batch_df)}</div>
                </div>
                """, unsafe_allow_html=True)
                kpi2.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">High Risk Identified</div>
                    <div class="metric-value" style="color: #e11d48;">{high_count}</div>
                    <div class="metric-subtext" style="color: #e11d48;">{(high_count/len(batch_df)*100):.1f}% of cohort</div>
                </div>
                """, unsafe_allow_html=True)
                kpi3.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Low Risk Cohort</div>
                    <div class="metric-value" style="color: #16a34a;">{low_count}</div>
                    <div class="metric-subtext">{(low_count/len(batch_df)*100):.1f}% of cohort</div>
                </div>
                """, unsafe_allow_html=True)
                kpi4.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Cohort Mean Risk</div>
                    <div class="metric-value">{avg_risk:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Visual charts
                c_chart1, c_chart2 = st.columns(2)
                with c_chart1:
                    fig_pie = px.pie(
                        names=["Low Risk (Healthy)", "High Risk (Disease)"],
                        values=[low_count, high_count],
                        color=["Low Risk (Healthy)", "High Risk (Disease)"],
                        color_discrete_map={"Low Risk (Healthy)": "#22c55e", "High Risk (Disease)": "#f43f5e"},
                        title="Cohort Risk Stratification Distribution",
                        hole=0.45
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with c_chart2:
                    fig_hist = px.histogram(
                        results_df,
                        x="Risk_Probability_%",
                        nbins=15,
                        color="Predicted_Risk",
                        color_discrete_map={"Low Risk (Healthy)": "#22c55e", "High Risk (Disease Detected)": "#f43f5e"},
                        title="Predicted Risk Probability Distribution"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Detailed Table
                st.markdown("#### 📋 Patient Screening Details")
                st.dataframe(
                    results_df.style.applymap(
                        lambda val: "background-color: #fee2e2; color: #991b1b; font-weight: bold;" if "High Risk" in str(val) or "🔴" in str(val)
                        else ("background-color: #dcfce7; color: #166534;" if "Low Risk" in str(val) or "🟢" in str(val) else ""),
                        subset=['Predicted_Risk', 'Triage_Priority']
                    ),
                    use_container_width=True
                )
                
                # Export Results
                csv_export = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Complete Batch Predictions (CSV)",
                    data=csv_export,
                    file_name="cardiopredict_batch_results.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ---------------------------------------------------------
# TAB 4: Model Benchmark & XAI
# ---------------------------------------------------------
elif navigation == "📊 Model Benchmarks & XAI":
    st.markdown('<div class="section-title">📊 Model Benchmark & Explainable AI (XAI)</div>', unsafe_allow_html=True)
    st.write("Compare the 3 models trained and tuned in the Google Colab experiment (`Random Forest`, `Logistic Regression`, and `Support Vector Machine`).")
    
    # Side-by-side metric comparison table
    model_names = ["Random Forest", "Logistic Regression", "Support Vector Machine"]
    comparison_data = []
    for m_name in model_names:
        m = metrics_data[m_name]
        comparison_data.append({
            "Model Name": m_name,
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "Precision (Macro)": f"{m['precision']*100:.2f}%",
            "Recall (Macro)": f"{m['recall']*100:.2f}%",
            "F1-Score (Macro)": f"{m['f1']*100:.2f}%"
        })
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df.set_index("Model Name"), use_container_width=True)
    
    col_bench1, col_bench2 = st.columns(2)
    
    with col_bench1:
        st.markdown(f"#### 🎯 Confusion Matrix: {selected_model_name}")
        cm = np.array(metrics_data[selected_model_name]['confusion_matrix'])
        labels = ["Healthy (0)", "Disease (1)"]
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=labels,
            y=labels,
            labels=dict(x="Predicted Diagnosis", y="Actual Condition", color="Count"),
            color_continuous_scale="Blues" if selected_model_name == "Random Forest" else "Greens"
        )
        fig_cm.update_layout(height=360)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_bench2:
        st.markdown("#### 🌟 Feature Importance (Random Forest Gini Weight)")
        rf_feat = metrics_data["Random Forest"].get("feature_importances", {})
        feat_df = pd.DataFrame(list(rf_feat.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
        
        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Random Forest Relative Feature Importances"
        )
        fig_feat.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_feat, use_container_width=True)
        
    # Detailed Classification Report
    st.markdown(f"#### 📄 Full Classification Report: {selected_model_name}")
    rep = metrics_data[selected_model_name]['report']
    rep_df = pd.DataFrame(rep).transpose()
    st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)

# ---------------------------------------------------------
# TAB 5: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
elif navigation == "📈 Exploratory Data Analysis (EDA)":
    st.markdown('<div class="section-title">📈 Exploratory Data Analysis & Population Insights</div>', unsafe_allow_html=True)
    st.write("Interactive visualizations replicating the exploratory analysis from the project notebook.")
    
    if df_data is not None:
        eda1, eda2 = st.columns(2)
        
        with eda1:
            st.markdown("#### 🎯 Target Balance (Heart Disease vs Healthy)")
            counts = df_data['target'].value_counts().reset_index()
            counts.columns = ['Status', 'Count']
            counts['Condition'] = counts['Status'].map({0: 'Healthy (0)', 1: 'Heart Disease (1)'})
            
            fig_target = px.bar(
                counts,
                x='Condition',
                y='Count',
                color='Condition',
                color_discrete_map={'Healthy (0)': '#22c55e', 'Heart Disease (1)': '#ef4444'},
                title="Target Variable Distribution"
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
        with eda2:
            st.markdown("#### 🫀 Chest Pain Type vs Heart Disease")
            cp_summary = df_data.groupby(['cp', 'target']).size().reset_index(name='count')
            cp_summary['Chest Pain'] = cp_summary['cp'].map({
                0: '0: Typical Angina',
                1: '1: Atypical Angina',
                2: '2: Non-Anginal',
                3: '3: Asymptomatic'
            })
            cp_summary['Condition'] = cp_summary['target'].map({0: 'Healthy', 1: 'Heart Disease'})
            
            fig_cp = px.bar(
                cp_summary,
                x='Chest Pain',
                y='count',
                color='Condition',
                barmode='group',
                color_discrete_map={'Healthy': '#22c55e', 'Heart Disease': '#ef4444'},
                title="Incidence by Chest Pain Category"
            )
            st.plotly_chart(fig_cp, use_container_width=True)
            
        eda3, eda4 = st.columns(2)
        with eda3:
            st.markdown("#### 📈 Age vs Maximum Heart Rate (thalach)")
            try:
                fig_scatter = px.scatter(
                    df_data,
                    x='age',
                    y='thalach',
                    color=df_data['target'].map({0: 'Healthy', 1: 'Heart Disease'}),
                    color_discrete_map={'Healthy': '#22c55e', 'Heart Disease': '#ef4444'},
                    trendline="ols",
                    title="Age vs Max Heart Rate with Trendline",
                    labels={'age': 'Patient Age (years)', 'thalach': 'Max Heart Rate Achieved (bpm)'}
                )
            except Exception:
                fig_scatter = px.scatter(
                    df_data,
                    x='age',
                    y='thalach',
                    color=df_data['target'].map({0: 'Healthy', 1: 'Heart Disease'}),
                    color_discrete_map={'Healthy': '#22c55e', 'Heart Disease': '#ef4444'},
                    title="Age vs Max Heart Rate",
                    labels={'age': 'Patient Age (years)', 'thalach': 'Max Heart Rate Achieved (bpm)'}
                )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with eda4:
            st.markdown("#### 🩸 Serum Cholesterol Distribution")
            fig_box = px.box(
                df_data,
                x=df_data['target'].map({0: 'Healthy', 1: 'Heart Disease'}),
                y='chol',
                color=df_data['target'].map({0: 'Healthy', 1: 'Heart Disease'}),
                color_discrete_map={'Healthy': '#22c55e', 'Heart Disease': '#ef4444'},
                title="Cholesterol Levels Across Target Groups",
                labels={'x': 'Cardiac Condition', 'chol': 'Serum Cholesterol (mg/dl)'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Correlation Heatmap
        st.markdown("#### 🗺️ Correlation Matrix of All Clinical Attributes")
        corr = df_data.corr().round(2)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------
# TAB 6: Clinical Dictionary & Info
# ---------------------------------------------------------
elif navigation == "📖 Clinical Dictionary & Info":
    st.markdown('<div class="section-title">📖 Clinical Dictionary & Project Methodology</div>', unsafe_allow_html=True)
    st.write("Comprehensive guide explaining each clinical parameter, medical threshold, and experimental pipeline details.")
    
    st.markdown("""
    ### 🔬 Clinical Feature Definitions

    | Feature Code | Medical Name | Clinical Description & Diagnostic Ranges |
    | :--- | :--- | :--- |
    | **`age`** | Age | Age of the individual in years. |
    | **`sex`** | Gender | Binary biological sex (`1 = Male`, `0 = Female`). |
    | **`cp`** | Chest Pain Type | Categorized into 4 presentations: `0 = Typical Angina`, `1 = Atypical Angina`, `2 = Non-Anginal Pain`, `3 = Asymptomatic`. |
    | **`trestbps`** | Resting Blood Pressure | Resting BP measured in mm Hg upon hospital admission. Desirable: `< 120 mm Hg`. Hypertension: `≥ 130 mm Hg`. |
    | **`chol`** | Serum Cholesterol | Total serum cholesterol in mg/dl. Desirable: `< 200 mg/dl`. High: `> 240 mg/dl`. |
    | **`fbs`** | Fasting Blood Sugar | Indicator whether fasting blood sugar is `> 120 mg/dl` (`1 = True`, `0 = False`). |
    | **`restecg`** | Resting ECG | Resting electrocardiogram findings: `0 = Normal`, `1 = ST-T wave abnormality`, `2 = Left ventricular hypertrophy`. |
    | **`thalach`** | Max Heart Rate | Maximum heart rate (bpm) achieved during maximal treadmill exercise stress test. |
    | **`exang`** | Exercise-Induced Angina | Whether angina (chest pain) occurred during exercise (`1 = Yes`, `0 = No`). |
    | **`oldpeak`** | ST Depression | ST segment depression induced by exercise relative to rest (indicates myocardial ischemia). |
    | **`slope`** | Peak ST Segment Slope | Slope of the peak exercise ST segment: `0 = Upsloping`, `1 = Flat`, `2 = Downsloping`. |
    | **`ca`** | Major Fluoroscopy Vessels | Number of major coronary vessels (`0 to 4`) visualized under fluoroscopy. Higher counts indicate significant calcification. |
    | **`thal`** | Thalassemia / Nuclear Scan | Thallium stress scintigraphy results: `1 = Normal`, `2 = Fixed defect`, `3 = Reversible defect`. |
    | **`target`** | Heart Disease Status | Primary target label: `0 = Absence of heart disease (< 50% stenosis)`, `1 = Presence of heart disease (> 50% diameter narrowing)`. |

    ---

    ### 🏗️ Machine Learning Methodology
    - **Data Pipeline**: Cleaned UCI / Kaggle dataset. Dropped 723 duplicate records to eliminate data leakage, resulting in 302 unique patients.
    - **Feature Preprocessing**: Categorical encoding and consistent train/test partitioning (`test_size=0.2, random_state=42`).
    - **Models Evaluated**:
      1. **Random Forest Classifier**: Tuned with 200 estimators, maximum tree depth of 5 (`83.6% accuracy`).
      2. **Logistic Regression**: Tuned with `C=1.0` and `liblinear` solver (`78.7% accuracy`).
      3. **Support Vector Machine**: Tuned with `C=10.0` and linear kernel with Platt calibration (`78.7% accuracy`).
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
    CardioPredict AI • Built with Streamlit, Scikit-Learn & Plotly • Inspired by Student Dropout Analytics Architecture
</div>
""", unsafe_allow_html=True)
