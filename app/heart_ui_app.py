import streamlit as st
import numpy as np
import pandas as pd
import joblib
import traceback
import plotly.graph_objects as go
import os
import json
from PIL import Image

import auth
import db  # unified name for consistency

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# ---------------- Session State ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "email" not in st.session_state:
    st.session_state.email = ""
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = None
if "age" not in st.session_state:
    st.session_state.age = None
if "gender" not in st.session_state:
    st.session_state.gender = None

# ---------------- Load Model (Fusion + XGB fallback) ----------------
def load_model_pipeline():
    """
    Returns:
      model: the final model object to call predict_proba on (fusion or single pipeline)
      loaded: the raw loaded object (could be a dict with base models + fusion)
      metrics: dict loaded from assets/model_metrics_extended.json if present
      errors: list of error messages encountered while loading
    """
    model, loaded, errors = None, None, []
    metrics = {}
    try:
        # 🩺 Highest priority: Symptom+Clinical fusion model
        if os.path.exists("models/symptom_model.joblib"):
            loaded = joblib.load("models/symptom_model.joblib")
            if isinstance(loaded, dict) and "fusion" in loaded:
                model = loaded["fusion"]
            else:
                model = loaded

        # ❤️ Next priority: heart_fusion_v2
        elif os.path.exists("models/heart_fusion_v2.joblib"):
            loaded = joblib.load("models/heart_fusion_v2.joblib")
            if isinstance(loaded, dict) and "fusion" in loaded:
                model = loaded["fusion"]
            else:
                model = loaded

        # 🧠 Final fallback: XGBoost baseline
        elif os.path.exists("models/heart_xgb.joblib"):
            loaded = joblib.load("models/heart_xgb.joblib")
            model = loaded

        # Load metrics if present
        if os.path.exists("assets/model_metrics_extended.json"):
            with open("assets/model_metrics_extended.json", "r") as f:
                metrics = json.load(f)
    except Exception as e:
        errors.append("model load error:\n" + "".join(traceback.format_exception_only(type(e), e)))
    return model, loaded, metrics, errors

model, loaded, metrics, load_errors = load_model_pipeline()
MODEL_AVAILABLE = model is not None

if not MODEL_AVAILABLE:
    st.warning("Model failed to load. Prediction features will be disabled.")
    for err in load_errors:
        st.caption(err.replace("\n", " | ")[:1000])

# ---------------- helper functions ----------------
def categorize_age(a):
    try:
        a = int(a)
    except Exception:
        return "Unknown"
    if a < 25: return "18-24"
    if a < 30: return "25-29"
    if a < 35: return "30-34"
    if a < 40: return "35-39"
    if a < 45: return "40-44"
    if a < 50: return "45-49"
    if a < 55: return "50-54"
    if a < 60: return "55-59"
    if a < 65: return "60-64"
    if a < 70: return "65-69"
    if a < 75: return "70-74"
    if a < 80: return "75-79"
    return "80 or older"

def yn_to_int(v):
    return 1 if str(v).lower() in {"yes","y","true","1"} else 0

def get_expected_feature_names_from_pipeline(pipeline_obj):
    # Try multiple strategies to discover the expected input feature names for a pipeline
    try:
        if hasattr(pipeline_obj, "feature_names_in_"):
            return list(pipeline_obj.feature_names_in_)
    except Exception:
        pass
    try:
        # If it's a full pipeline with a ColumnTransformer named 'preprocessor'
        if hasattr(pipeline_obj, "named_steps") and "preprocessor" in pipeline_obj.named_steps:
            pre = pipeline_obj.named_steps["preprocessor"]
            try:
                return list(pre.get_feature_names_out())
            except Exception:
                names = []
                if hasattr(pre, "transformers_"):
                    for name, trans, cols in pre.transformers_:
                        if trans is None:
                            continue
                        try:
                            if hasattr(trans, "get_feature_names_out"):
                                out = list(trans.get_feature_names_out(cols))
                                names.extend(out)
                            else:
                                if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
                                    ohe = trans.named_steps["onehot"]
                                    if hasattr(ohe, "get_feature_names_out"):
                                        out = list(ohe.get_feature_names_out(cols))
                                        names.extend(out)
                                    else:
                                        names.extend(cols if isinstance(cols, (list,tuple)) else [cols])
                                else:
                                    names.extend(cols if isinstance(cols, (list,tuple)) else [cols])
                        except Exception:
                            names.extend(cols if isinstance(cols, (list,tuple)) else [cols])
                if names:
                    return names
    except Exception:
        pass
    return None

# ---------------- Doctor Page ----------------
def doctor_page():
    st.markdown(
        "<h1 style='text-align: center; color: red;'>❤️ Doctor Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.write(f"Welcome Dr. **{st.session_state.username}** 👩‍⚕️👨‍⚕️")

    patient_name = st.text_input("👤 Enter Patient's Name")

    # Label dictionaries
    sex_labels = {0: "Female", 1: "Male"}
    cp_labels = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
    restecg_labels = {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}
    slope_labels = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    thal_labels = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}

    st.sidebar.header("🩺 Patient Input Features")

    # Collect inputs (basic: age + sex)
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex_choice = st.sidebar.radio("Sex", ["Female", "Male"])
    sex = 1 if sex_choice == "Male" else 0

    # ---------------- extra symptom/lifestyle fields (placed right after Age/Sex) ----------------
    # These are shown to the doctor immediately after age/sex (as requested).
    # They will be included when building df_full for single-pipeline models.
    BMI = st.sidebar.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    Smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
    AlcoholDrinking = st.sidebar.selectbox("Alcohol Drinking", ["No", "Yes"])
    Stroke = st.sidebar.selectbox("History of Stroke", ["No", "Yes"])
    PhysicalHealth = st.sidebar.number_input("Physical Health (days poor past 30)", 0, 30, 0)
    MentalHealth = st.sidebar.number_input("Mental Health (days poor past 30)", 0, 30, 0)
    DiffWalking = st.sidebar.selectbox("Difficulty Walking", ["No", "Yes"])
    PhysicalActivity = st.sidebar.selectbox("Physical Activity", ["No", "Yes"])
    SleepTime = st.sidebar.number_input("Avg Sleep Time (hrs)", 0, 24, 7)
    Asthma = st.sidebar.selectbox("Asthma", ["No", "Yes"])
    KidneyDisease = st.sidebar.selectbox("Kidney Disease", ["No", "Yes"])
    SkinCancer = st.sidebar.selectbox("Skin Cancer", ["No", "Yes"])
    Race = st.sidebar.selectbox("Race", ["White", "Black", "Asian", "Other"])
    Diabetic = st.sidebar.selectbox("Diabetic", ["No", "Yes", "During pregnancy", "Borderline"])

    # Add GenHealth as requested (placed near the basic demographic inputs)
    GenHealth = st.sidebar.selectbox("General Health (GenHealth)", ["Excellent", "Very good", "Good", "Fair", "Poor"])

    # divider to visually separate symptom inputs from clinical inputs
    st.sidebar.markdown("---")

    # Clinical inputs (kept after the symptom inputs) - converted to input boxes / selectboxes for unified style
    cp_choice = st.sidebar.selectbox("Chest Pain Type", list(cp_labels.values()))
    cp = [k for k, v in cp_labels.items() if v == cp_choice][0]
    trestbps = st.sidebar.number_input("Resting BP (mm Hg)", min_value=50, max_value=300, value=120, step=1)
    chol = st.sidebar.number_input("Cholesterol (mg/dl)", min_value=50, max_value=1000, value=200, step=1)
    fbs_choice = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 1 if fbs_choice == "Yes" else 0
    recg_choice = st.sidebar.selectbox("Resting ECG", list(restecg_labels.values()))
    restecg = [k for k, v in restecg_labels.items() if v == recg_choice][0]
    thalach = st.sidebar.number_input("Max Heart Rate", min_value=30, max_value=300, value=150, step=1)
    exang_choice = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang_choice == "Yes" else 0
    oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f")
    slope_choice = st.sidebar.selectbox("Slope of ST Segment", list(slope_labels.values()))
    slope = [k for k, v in slope_labels.items() if v == slope_choice][0]
    ca = st.sidebar.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, step=1)
    thal_choice = st.sidebar.selectbox("Thalassemia", list(thal_labels.values()))
    thal = [k for k, v in thal_labels.items() if v == thal_choice][0]

    model_data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    display_data = {
        "Age": age,
        "Sex": sex_labels[sex],
        "Chest Pain Type": cp_labels[cp],
        "Resting BP (mm Hg)": trestbps,
        "Cholesterol (mg/dl)": chol,
        "Fasting Blood Sugar": "Yes" if fbs == 1 else "No",
        "Resting ECG": restecg_labels[restecg],
        "Max Heart Rate": thalach,
        "Exercise Induced Angina": "Yes" if exang == 1 else "No",
        "ST Depression": oldpeak,
        "Slope of ST Segment": slope_labels[slope],
        "Major Vessels": ca,
        "Thalassemia": thal_labels[thal]
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("👤 Patient Data")
        df_display = pd.DataFrame([display_data]).T.rename(columns={0: "Value"}).astype(str)
        st.dataframe(df_display, width='stretch', height=min(600, len(df_display) * 40))

    with col2:
        if st.button("🔍 Predict"):
            if not patient_name:
                st.error("⚠️ Please enter the patient's name before predicting.")
            elif not MODEL_AVAILABLE:
                st.error("Model not loaded — cannot predict. Check console/errors above.")
            else:
                try:
                    df_input = pd.DataFrame([model_data])

                    # If we loaded a dict bundle earlier, use base models to form fusion input
                    if isinstance(loaded, dict):
                        try:
                            xgb_model = loaded.get("xgb")
                            rf_model = loaded.get("rf")
                            ann_model = loaded.get("ann_base")
                            fusion_model = loaded.get("fusion", model)
                        except Exception as e:
                            xgb_model = rf_model = ann_model = fusion_model = None

                        if any(m is None for m in [xgb_model, rf_model, ann_model, fusion_model]):
                            st.error("Fusion or base models missing from bundle. Prediction cannot proceed.")
                            st.stop()

                        # Convert to numpy array for base models (they likely expect raw numeric features)
                        X_np = np.array(df_input, dtype=float)
                        xgb_prob = float(xgb_model.predict_proba(X_np)[0][1])
                        rf_prob = float(rf_model.predict_proba(X_np)[0][1])
                        ann_prob = float(ann_model.predict_proba(X_np)[0][1])

                        fusion_input = np.array([[xgb_prob, rf_prob, ann_prob]], dtype=float)
                        prob = float(fusion_model.predict_proba(fusion_input)[0][1])

                    else:
                        # ---------- SINGLE-PIPELINE / COMBINED MODEL FLOW ----------
                        # Build extras (symptom/lifestyle) and derive AgeCategory & Sex string
                        extra = {
                            "BMI": float(BMI),
                            "Smoking": yn_to_int(Smoking),
                            "AlcoholDrinking": yn_to_int(AlcoholDrinking),
                            "Stroke": yn_to_int(Stroke),
                            "PhysicalHealth": float(PhysicalHealth),
                            "MentalHealth": float(MentalHealth),
                            "DiffWalking": yn_to_int(DiffWalking),
                            "PhysicalActivity": yn_to_int(PhysicalActivity),
                            "SleepTime": float(SleepTime),
                            "Asthma": yn_to_int(Asthma),
                            "KidneyDisease": yn_to_int(KidneyDisease),
                            "SkinCancer": yn_to_int(SkinCancer),
                            "Race": str(Race),
                            "Diabetic": str(Diabetic),
                        }
                        extra["AgeCategory"] = categorize_age(age)
                        # include Sex string as many symptom datasets use "Sex"
                        extra["Sex"] = "Male" if sex == 1 else "Female"
                        # include GenHealth (string)
                        extra["GenHealth"] = str(GenHealth)
                        # include defaults for HeartDisease and source (prevents missing columns errors)
                        # HeartDisease default = 0 (silent)
                        extra["HeartDisease"] = 0
                        # source default = "clinical" (silent)
                        extra["source"] = "clinical"

                        # Merge clinical model_data with extras to form a superset
                        df_full = pd.DataFrame([ {**model_data, **extra} ])

                        # Discover expected feature names for pipeline
                        expected_cols = get_expected_feature_names_from_pipeline(model)

                        if expected_cols:
                            # check missing
                            missing = [c for c in expected_cols if c not in df_full.columns]
                            if missing:
                                st.error("Model expects additional inputs that are not present in the UI.")
                                st.write("Missing columns needed by model:", missing)
                                st.info("Add these fields to the doctor sidebar or retrain the model with fewer required features.")
                                st.stop()
                            # reorder df_full to expected (most pipelines will accept pandas DF)
                            try:
                                df_full = df_full[expected_cols]
                            except Exception:
                                # if reordering fails, let predict_proba attempt to handle it; but keep user informed
                                st.info("Warning: failed to reorder columns to pipeline expectation; will attempt prediction anyway.")
                        else:
                            # No expected cols discovered — attempt to call predict_proba with df_full directly,
                            # but ensure column names match what we can supply
                            pass

                        # final predict
                        prob = float(model.predict_proba(df_full)[0][1])

                    # Save prediction
                    db.save_prediction(st.session_state.email, patient_name, prob, "High" if prob > 0.5 else "Low", model_data)

                    # AI Diagnosis Section
                    st.markdown("---")
                    st.subheader("🧠 AI Diagnosis Result")
                    if prob <= 0.30:
                        st.success(f"✅ Low Risk ({prob:.2%})")
                        st.caption("No immediate concern detected.")
                    elif prob <= 0.60:
                        st.warning(f"⚠️ Moderate Risk ({prob:.2%})")
                        st.caption("Recommend lifestyle checks and further evaluation.")
                    elif prob <= 0.85:
                        st.error(f"🚨 High Risk ({prob:.2%})")
                        st.caption("Immediate medical attention recommended.")
                    else:
                        st.error(f"🩸 Critical Risk ({prob:.2%})")
                        st.caption("Severe indicators of heart disease risk detected!")

                    if isinstance(metrics, dict) and "Top SHAP Features" in metrics:
                        top_feats = metrics["Top SHAP Features"]
                        st.write("**Key Contributing Factors:**")
                        for feat in top_feats:
                            st.write(f"- {feat['feature']} (impact: {feat['impact']})")

                    if isinstance(metrics, dict) and metrics:
                        st.markdown("---")
                        st.subheader("📊 Model Performance Summary")
                        cols = st.columns(3)
                        fusion_metrics = metrics.get("Symptom+Clinical Fusion", metrics)
                        train_acc = fusion_metrics.get("train_accuracy", fusion_metrics.get("Train Accuracy", 0))
                        val_acc = fusion_metrics.get("val_accuracy", fusion_metrics.get("Validation Accuracy", 0))
                        test_acc = fusion_metrics.get("test_accuracy", fusion_metrics.get("Test Accuracy", 0))
                        cols[0].metric("Train Accuracy", f"{train_acc:.2f}")
                        cols[1].metric("Validation Accuracy", f"{val_acc:.2f}")
                        cols[2].metric("Test Accuracy", f"{test_acc:.2f}")

                        st.markdown("### 🔍 Model Visual Insights")
                        plot_files = [
                            "roc_curve.png", "pr_curve.png", "feature_importance_bar.png",
                            "shap_summary.png", "correlation_heatmap.png", "top_features_risk.png",
                            "symptom_clinical_feature_importance.png"
                        ]
                        for pf in plot_files:
                            path = os.path.join("assets", pf)
                            if os.path.exists(path):
                                st.image(Image.open(path), caption=pf.replace(".png", "").replace("_", " ").title(), width='stretch')

                    fig = go.Figure(data=[go.Pie(labels=["Risk", "Safe"], values=[prob, 1 - prob],
                                                 hole=0.6, marker=dict(colors=["#ff4d4d", "#4CAF50"]))])
                    fig.update_traces(textinfo="percent+label", textfont_size=14)
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error("Prediction failed. See console for details.")
                    st.exception(e)

    st.subheader("📜 Prediction History")
    history = db.load_history(st.session_state.email)
    if history:
        hist_df = pd.DataFrame(history, columns=["DoctorEmail", "Patient", "Timestamp", "Probability", "Result", "Inputs"])
        st.dataframe(hist_df.drop(columns=["DoctorEmail"]), width='stretch')
    else:
        st.info("No predictions yet.")

# ---------------- Patient Page ----------------
def patient_page():
    st.markdown(
        "<h1 style='text-align: center; color: green;'>🧑‍🤝‍🧑 Patient Symptom Checker</h1>",
        unsafe_allow_html=True,
    )
    st.write(f"Hello **{st.session_state.username}** 👋, please answer a few questions.")

    st.sidebar.header("🩺 Basic Symptom Questions")

    chest_pain = st.sidebar.radio("Do you feel chest discomfort?", ["No", "Mild", "Severe"])
    breath = st.sidebar.radio("Do you experience shortness of breath?", ["No", "Sometimes", "Often"])
    fatigue = st.sidebar.radio("Do you often feel unusually tired?", ["No", "Yes"])
    dizziness = st.sidebar.radio("Do you experience dizziness or fainting?", ["No", "Yes"])
    family_history = st.sidebar.radio("Family history of heart disease?", ["No", "Yes"])

    risk_score = 0
    if chest_pain != "No": risk_score += 2
    if breath == "Sometimes": risk_score += 1
    if breath == "Often": risk_score += 2
    if fatigue == "Yes": risk_score += 1
    if dizziness == "Yes": risk_score += 2
    if family_history == "Yes": risk_score += 1

    if st.button("🩺 Check My Risk"):
        if risk_score >= 5:
            st.error("⚠️ Your symptoms suggest you should visit a doctor soon.")
        elif risk_score >= 3:
            st.warning("⚠️ Some symptoms detected. Consider getting a check-up.")
        else:
            st.success("✅ Your symptoms don’t show high risk, but stay healthy!")

# ---------------- Routing ----------------
if not st.session_state.logged_in:
    auth.login_screen()
else:
    if st.session_state.role == "doctor":
        doctor_page()
    else:
        patient_page()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.email = ""
        st.session_state.username = ""
        st.session_state.role = None
        st.session_state.age = None
        st.session_state.gender = None
