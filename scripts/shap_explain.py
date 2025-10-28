# scripts/shap_explain.py
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split

# Optional: silence some non-critical warnings (Convergence/Future)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure directories
os.makedirs("assets", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Paths ---
data_path = "data/heart.csv"
model_path = "models/heart_fusion_v2.joblib"
metrics_path = "assets/model_metrics_extended.json"

print("Loading data and model for SHAP analysis...")
# Load data
df = pd.read_csv(data_path)

# Load model bundle (expects keys: preprocessor, xgb, rf, ann_base, fusion)
model_bundle = joblib.load(model_path)

preprocessor = model_bundle["preprocessor"]
xgb = model_bundle["xgb"]
rf = model_bundle["rf"]
ann_base = model_bundle["ann_base"]
fusion = model_bundle["fusion"]

# Prepare validation set (representative sample for explanations)
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Preprocess validation features (same preprocessor used during training)
X_val_prep = preprocessor.transform(X_val)

# Helper to stack base-model probabilities into fusion inputs
def stack_probs(models, X_prepared):
    # each model.predict_proba(X_prepared)[:,1] -> shape (n_samples,)
    return np.column_stack([m.predict_proba(X_prepared)[:, 1] for m in models])

base_models = [xgb, rf, ann_base]
X_val_stack = stack_probs(base_models, X_val_prep)  # shape (n_samples, 3)

# --- Compute SHAP values for the fusion model ---
print("Computing SHAP values...")
# Explainer: explain the fusion model's predict_proba function using the stacked inputs
explainer = shap.Explainer(fusion.predict_proba, X_val_stack)
shap_values = explainer(X_val_stack)  # shap_values.values shape: (n_samples, n_features, n_outputs)

# Feature names for the fusion inputs
feature_names = ["XGBoost_Prob", "RandomForest_Prob", "ANN_Prob"]
X_val_df = pd.DataFrame(X_val_stack, columns=feature_names)

# --- Global SHAP summary (bar) ---
print("Generating SHAP summary plot...")

# Ensure shap_values has the expected dims and pick the positive class (index 1)
# shap_values.values: shape (n_samples, n_features, n_outputs)
if shap_values.values.ndim == 3 and shap_values.values.shape[2] >= 2:
    # use class-1 (disease) SHAP values
    class_idx = 1
    mean_abs_shap = np.mean(np.abs(shap_values.values[:, :, class_idx]), axis=0).flatten()
else:
    # fallback (single-output)
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0).flatten()

# Build importance DataFrame
shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean |SHAP|": mean_abs_shap
}).sort_values(by="Mean |SHAP|", ascending=False)

# Plot (use seaborn; avoid deprecated palette usage warnings by not supplying hue)
plt.figure(figsize=(6, 3 + 0.5 * len(feature_names)))
sns.barplot(data=shap_importance, x="Mean |SHAP|", y="Feature")
plt.title("Global Feature Importance (Fusion Model)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("assets/shap_summary.png", bbox_inches="tight")
plt.close()

# --- Individual SHAP force/decision plot for one sample ---
sample_index = 10
if sample_index >= X_val_stack.shape[0]:
    sample_index = 0  # fallback if dataset is small

print(f"Generating SHAP force plot for sample index {sample_index}...")

# handle SHAP expected values across versions
expected_vals = None
if hasattr(explainer, "expected_values"):
    expected_vals = explainer.expected_values
elif hasattr(explainer, "expected_value"):
    expected_vals = explainer.expected_value

# Determine base value for the positive class (if available)
if expected_vals is None:
    base_value = None
elif isinstance(expected_vals, (list, np.ndarray)):
    # If multi-output, prefer index 1 (positive class) if present
    try:
        base_value = expected_vals[1] if len(expected_vals) > 1 else expected_vals[0]
    except Exception:
        base_value = expected_vals[0]
else:
    base_value = expected_vals

# Prepare shap values for the sample: choose class 1 if present
if shap_values.values.ndim == 3 and shap_values.values.shape[2] >= 2:
    sample_shap_vals = shap_values.values[sample_index, :, 1]
else:
    sample_shap_vals = shap_values.values[sample_index, :]

# >>> FIX START: ensure a valid numeric base_value and call force_plot directly (no fallback)
# Compute explicit base value (mean predicted probability for the positive class)
# This guarantees base_value is a float and avoids None issues in SHAP internals.
explicit_base = float(np.mean(fusion.predict_proba(X_val_stack)[:, 1]))

# Patient prediction for labeling
patient_pred = float(fusion.predict_proba(X_val_stack)[sample_index, 1])

# Generate force plot using explicit base value and features DataFrame
shap.force_plot(
    explicit_base,
    sample_shap_vals,
    features=X_val_df.iloc[sample_index, :],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title(f"SHAP Force Plot – Sample {sample_index}\nPredicted Heart Disease Risk: {patient_pred:.2%}")
plt.tight_layout()
plt.savefig("assets/shap_force_patient.png", bbox_inches="tight")
plt.close()
# >>> FIX END

# --- Save top features info to metrics JSON ---
# Recompute mean_abs_shap for JSON (use same class logic)
if shap_values.values.ndim == 3 and shap_values.values.shape[2] >= 2:
    mean_abs_shap = np.mean(np.abs(shap_values.values[:, :, 1]), axis=0).flatten()
else:
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0).flatten()

top_features = [
    {"feature": f, "impact": float(round(v, 6))}
    for f, v in sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
]

# Load existing metrics JSON if present
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as fh:
        try:
            metrics_data = json.load(fh)
        except Exception:
            metrics_data = {}
else:
    metrics_data = {}

metrics_data["Top SHAP Features"] = top_features

with open(metrics_path, "w") as fh:
    json.dump(metrics_data, fh, indent=4)

print("✅ SHAP analysis complete. Plots and updated metrics saved in /assets/")
