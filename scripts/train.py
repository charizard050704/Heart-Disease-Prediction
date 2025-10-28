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
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    log_loss,
    brier_score_loss
)
from xgboost import XGBClassifier

# Ensure directories
os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- Load dataset ---
uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
colnames = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

print("Downloading Cleveland dataset from UCI ...")
df = pd.read_csv(uci_url, names=colnames, na_values="?")
df = df.dropna().reset_index(drop=True)
df["target"] = (df["target"] > 0).astype(int)
df.to_csv("data/heart.csv", index=False)
print(f"Dataset cleaned and saved to data/heart.csv ({len(df)} rows).")

# --- Preprocessing ---
X = df.drop("target", axis=1)
y = df["target"]
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = list(set(X.columns) - set(num_features))
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# --- Split ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# --- Model ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    ))
])

print("Training XGBoost model...")
model.fit(X_train, y_train)
joblib.dump(model, "models/heart_xgb.joblib")

# --- Evaluation helper ---
def evaluate(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_proba)),
        "mae": mean_absolute_error(y, y_proba),
        "brier": brier_score_loss(y, y_proba),
        "logloss": log_loss(y, y_proba)
    }

# --- Evaluate splits ---
train_eval = evaluate(model, X_train, y_train)
val_eval = evaluate(model, X_val, y_val)
test_eval = evaluate(model, X_test, y_test)

# --- Terminal summary (clean) ---
print(f"Train Accuracy: {train_eval['accuracy']:.4f}")
print(f"Validation Accuracy: {val_eval['accuracy']:.4f}")
print(f"Test Accuracy: {test_eval['accuracy']:.4f}")
print("✅ Training complete. Model + metrics + plots saved.")

# --- Extended metrics file ---
metrics_summary = {
    "Train Accuracy": round(train_eval["accuracy"], 4),
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

# --- Visual assets ---
# Confusion Matrix
cm = confusion_matrix(y_val, val_eval["y_pred"])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Validation)")
plt.tight_layout()
plt.savefig("assets/confusion_matrix_overlay.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, val_eval["y_proba"])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend()
plt.tight_layout()
plt.savefig("assets/roc_curve.png")
plt.close()

# Precision–Recall Curve
prec, rec, _ = precision_recall_curve(y_val, val_eval["y_proba"])
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Validation)")
plt.tight_layout()
plt.savefig("assets/pr_curve.png")
plt.close()

# Error Metrics Bar
plt.figure()
bars = ["RMSE","MAE","Brier","LogLoss"]
vals = [val_eval["rmse"], val_eval["mae"], val_eval["brier"], val_eval["logloss"]]
plt.bar(bars, vals)
plt.title("Error Metrics Overview")
plt.tight_layout()
plt.savefig("assets/error_metrics_bar.png")
plt.close()

# Error Distribution
residuals = y_val - val_eval["y_proba"]
plt.figure()
sns.histplot(residuals, bins=20, kde=True, color="crimson")
plt.title("Error Distribution (Residuals)")
plt.tight_layout()
plt.savefig("assets/error_distribution.png")
plt.close()

# Feature Importance
xgb_model = model.named_steps["classifier"]
importance = xgb_model.feature_importances_
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False)[:15]
plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig("assets/feature_importance_bar.png")
plt.close()
