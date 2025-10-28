import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, log_loss, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# --- Ensure directories ---
os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- Load dataset ---
uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
colnames = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]

print("Downloading Cleveland dataset from UCI ...")
df = pd.read_csv(uci_url, names=colnames, na_values="?").dropna().reset_index(drop=True)
df["target"] = (df["target"] > 0).astype(int)
df.to_csv("data/heart.csv", index=False)
print(f"Dataset cleaned and saved to data/heart.csv ({len(df)} rows).")

# --- Preprocessing ---
X = df.drop("target", axis=1)
y = df["target"]
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = list(set(X.columns) - set(num_features))
numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# --- Split data ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# --- Transform features ---
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep   = preprocessor.transform(X_val)
X_test_prep  = preprocessor.transform(X_test)

# --- Base models ---
print("Training base models...")
xgb = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss"
)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
ann_base = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu',
                         solver='adam', max_iter=1000, random_state=42)

xgb.fit(X_train_prep, y_train)
rf.fit(X_train_prep, y_train)
ann_base.fit(X_train_prep, y_train)

# --- Fusion dataset (stacked probabilities) ---
def stack_probs(models, X):
    return np.column_stack([m.predict_proba(X)[:,1] for m in models])

base_models = [xgb, rf, ann_base]
X_train_stack = stack_probs(base_models, X_train_prep)
X_val_stack   = stack_probs(base_models, X_val_prep)
X_test_stack  = stack_probs(base_models, X_test_prep)

# --- Fusion model ---
fusion = MLPClassifier(hidden_layer_sizes=(16,8), activation='relu',
                       solver='adam', max_iter=1000, random_state=42)
print("Training fusion ANN...")
fusion.fit(X_train_stack, y_train)

# --- Evaluation helper ---
def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_prob)),
        "mae": mean_absolute_error(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "logloss": log_loss(y_true, y_prob)
    }

# --- Evaluate on validation/test ---
y_val_prob  = fusion.predict_proba(X_val_stack)[:,1]
y_val_pred  = (y_val_prob > 0.5).astype(int)
y_test_prob = fusion.predict_proba(X_test_stack)[:,1]
y_test_pred = (y_test_prob > 0.5).astype(int)

val_eval  = evaluate(y_val, y_val_pred, y_val_prob)
test_eval = evaluate(y_test, y_test_pred, y_test_prob)

# --- Terminal summary ---
print(f"Train Accuracy: {fusion.score(X_train_stack, y_train):.4f}")
print(f"Validation Accuracy: {val_eval['accuracy']:.4f}")
print(f"Test Accuracy: {test_eval['accuracy']:.4f}")
print("✅ Fusion model training complete. All metrics and plots saved.")

# --- Save models ---
joblib.dump({
    "preprocessor": preprocessor,
    "xgb": xgb,
    "rf": rf,
    "ann_base": ann_base,
    "fusion": fusion
}, "models/heart_fusion_v2.joblib")

# --- Extended metrics JSON ---
metrics_summary = {
    "Train Accuracy": round(fusion.score(X_train_stack, y_train), 4),
    "Validation Accuracy": round(val_eval["accuracy"], 4),
    "Test Accuracy": round(test_eval["accuracy"], 4),
    "Accuracy": round(val_eval["accuracy"], 4),
    "Precision": round(val_eval["precision"], 4),
    "Recall": round(val_eval["recall"], 4),
    "F1-Score": round(val_eval["f1"], 4),
    "RMSE": round(val_eval["rmse"], 4),
    "MAE": round(val_eval["mae"], 4),
    "Brier Score": round(val_eval["brier"], 4),
    "Log Loss": round(val_eval["logloss"], 4)
}
with open("assets/model_metrics_extended.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)

# --- Visualizations ---
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Fusion - Validation)")
plt.tight_layout()
plt.savefig("assets/confusion_matrix_overlay.png")
plt.close()

# ROC & PR curves
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Fusion Validation)")
plt.legend()
plt.tight_layout()
plt.savefig("assets/roc_curve.png")
plt.close()

prec, rec, _ = precision_recall_curve(y_val, y_val_prob)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Fusion Validation)")
plt.tight_layout()
plt.savefig("assets/pr_curve.png")
plt.close()

# Error bars & distributions
plt.figure()
bars = ["RMSE","MAE","Brier","LogLoss"]
vals = [val_eval["rmse"], val_eval["mae"], val_eval["brier"], val_eval["logloss"]]
plt.bar(bars, vals)
plt.title("Fusion Model – Error Metrics Overview")
plt.tight_layout()
plt.savefig("assets/error_metrics_bar.png")
plt.close()

residuals = y_val - y_val_prob
plt.figure()
sns.histplot(residuals, bins=20, kde=True, color="crimson")
plt.title("Fusion Model – Error Distribution")
plt.tight_layout()
plt.savefig("assets/error_distribution.png")
plt.close()
