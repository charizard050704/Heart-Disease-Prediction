# scripts/symptom_ai.py
"""
Unified Symptom+Clinical training (no truncation).
- Uses both: data/heart.csv and data/heart_symptoms.csv
- Creates a union of all columns and fills missing ones with NaN
- Trains RandomForestClassifier with RandomizedSearchCV
- Outputs: models/symptom_model.joblib, plots, metrics JSON
"""
import matplotlib
matplotlib.use("Agg")
import os, json, joblib, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from scipy.stats import randint


warnings.filterwarnings("ignore")

DATA_CLINICAL = "data/heart.csv"
DATA_SYMPTOM = "data/heart_symptoms.csv"
MODEL_DIR = Path("models"); ASSETS_DIR = Path("assets")
METRICS_PATH = ASSETS_DIR / "model_metrics_extended.json"
MODEL_OUT = MODEL_DIR / "symptom_model.joblib"
# Skip retraining if model already exists
if MODEL_OUT.exists():
    print(f"⚡ Existing model found at {MODEL_OUT}. Skipping retraining...")
    model = joblib.load(MODEL_OUT)
    print("Loaded existing model successfully.")

    # Recreate plots & metrics quickly
    feat_imp = model.named_steps["clf"].feature_importances_
    feat_imp = np.array(feat_imp)
    feat_imp = np.sort(feat_imp)[::-1]

    plt.figure(figsize=(10, 6))
    top_n = min(40, len(feat_imp))
    plt.bar(range(top_n), feat_imp[:top_n])
    plt.title("Top Feature Importances – Symptom+Clinical Model (Reloaded)")
    plt.xlabel("Feature Rank")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR/"symptom_clinical_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("✅ Regenerated feature importance plot.")

    print("✅ Model reload complete. Exiting early.")
    exit(0)

MODEL_DIR.mkdir(parents=True, exist_ok=True); ASSETS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE, N_ITER_SEARCH, CV_FOLDS, TEST_SIZE = 42, 40, 3, 0.18

BASE_RF_PARAMS = dict(
    n_estimators=600, max_depth=30, min_samples_split=3, min_samples_leaf=1,
    class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
)

def load_csv(path): return pd.read_csv(Path(path))

print("Loading datasets...")
clin = load_csv(DATA_CLINICAL); symp = load_csv(DATA_SYMPTOM)
print(f"Clinical: {clin.shape} | Symptom: {symp.shape}")

# Identify target
target = None
for c in ["target","Target","HeartDisease","heart_disease"]:
    if c in clin.columns: target = c; break
if target is None: raise RuntimeError("No target column found in clinical data")

clin[target] = clin[target].apply(lambda x: 1 if str(x).lower() in {"1","yes","true"} else 0)
if "HeartDisease" in symp.columns:
    symp["target"] = symp["HeartDisease"].apply(lambda x: 1 if str(x).lower() in {"1","yes","true"} else 0)
else:
    symp["target"] = np.nan

clin["source"] = "clinical"; symp["source"] = "symptom"

# Create union of columns
all_cols = sorted(set(clin.columns) | set(symp.columns))
for df in [clin, symp]:
    for c in all_cols:
        if c not in df.columns:
            df[c] = np.nan

combined = pd.concat([clin[all_cols], symp[all_cols]], axis=0, ignore_index=True)
print("Combined shape:", combined.shape)

y = combined["target"].astype(float).fillna(0)
X = combined.drop(columns=["target"])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
prep = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)])

rf = RandomForestClassifier(**BASE_RF_PARAMS)
pipe = Pipeline([("pre", prep), ("clf", rf)])

params = {
    "clf__n_estimators": randint(300, 900),
    "clf__max_depth": [None] + list(range(20, 61, 10)),
    "clf__min_samples_split": randint(2, 8),
    "clf__min_samples_leaf": randint(1, 5)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=(y>0), random_state=RANDOM_STATE)
print("Train/test split:", X_train.shape, X_test.shape)

print("Running RandomizedSearchCV...")
start = time.time()
search = RandomizedSearchCV(pipe, param_distributions=params, n_iter=N_ITER_SEARCH,
                             scoring="roc_auc", cv=CV_FOLDS, n_jobs=-1, random_state=RANDOM_STATE)
search.fit(X_train, y_train)
best = search.best_estimator_
print(f"Search complete ({(time.time()-start)/60:.2f} min), best params:", search.best_params_)

y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:,1]
acc, roc = accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob)
print(f"Accuracy={acc:.4f}, ROC_AUC={roc:.4f}\n", classification_report(y_test, y_pred))

joblib.dump(best, MODEL_OUT)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig(ASSETS_DIR/"symptom_confusion_matrix.png"); plt.close()

feat_imp = best.named_steps["clf"].feature_importances_
feat_imp = np.array(feat_imp)
feat_imp = np.sort(feat_imp)[::-1]

plt.figure(figsize=(10, 6))
top_n = min(40, len(feat_imp))
plt.bar(range(top_n), feat_imp[:top_n])
plt.title("Top Feature Importances – Symptom+Clinical Model")
plt.xlabel("Feature Rank")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(ASSETS_DIR/"symptom_clinical_feature_importance.png", bbox_inches="tight")
plt.close()
print("✅ Saved feature importance plot successfully.")

metrics = {}
if METRICS_PATH.exists():
    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}
metrics["Symptom+Clinical Fusion"] = {
    "test_accuracy": float(acc), "roc_auc": float(roc),
    "train_rows": int(X_train.shape[0]), "test_rows": int(X_test.shape[0]),
    "features_total": int(X.shape[1]), "best_params": {k:str(v) for k,v in search.best_params_.items()}
}
json.dump(metrics, open(METRICS_PATH,"w"), indent=4)
print("✅ Training complete. Model + metrics saved.")
