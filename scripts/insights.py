# scripts/insights.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Ensure directories
os.makedirs("assets", exist_ok=True)
os.makedirs("data", exist_ok=True)

data_path = "data/heart.csv"
metrics_path = "assets/model_metrics_extended.json"

print("Loading dataset for correlation and insights...")
df = pd.read_csv(data_path)

# --- 1. Correlation Heatmap (PNG for UI) ---
print("Generating correlation heatmap (PNG for UI)...")
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap (Heart Disease Dataset)")
plt.tight_layout()
plt.savefig("assets/correlation_heatmap.png", bbox_inches="tight")
plt.close()

# --- 1B. Correlation Heatmap (EPS for Professor: clean, no title) ---
print("Generating correlation heatmap (EPS for report)...")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
# no title in EPS version
plt.tight_layout()
plt.savefig("assets/correlation_heatmap.eps", format='eps', bbox_inches="tight")
plt.close()

# --- 2. Top Features Driving Heart Risk (PNG for UI) ---
print("Generating top features bar chart (PNG for UI)...")
if "target" in df.columns:
    target_corr = corr["target"].drop("target").sort_values(
        key=lambda s: np.abs(s), ascending=False
    )
    top_corr = target_corr.head(10)
else:
    top_corr = pd.Series(dtype=float)

plt.figure(figsize=(7,5))
sns.barplot(x=top_corr.values, y=top_corr.index, hue=top_corr.index,
            palette="Reds_r", legend=False)
plt.ylabel("")
plt.title("Top Features Driving Heart Disease Risk")
plt.xlabel("Correlation with Target")
plt.tight_layout()
plt.savefig("assets/top_features_risk.png", bbox_inches="tight")
plt.close()

# --- 2B. EPS Version for Professor ---
print("Generating top features bar chart (EPS for report)...")
plt.figure(figsize=(7,5))
sns.barplot(x=top_corr.values, y=top_corr.index, hue=top_corr.index,
            palette="Reds_r", legend=False)
plt.ylabel("")
# no title for EPS
plt.xlabel("Correlation with Target")
plt.tight_layout()
plt.savefig("assets/top_features_risk.eps", format='eps', bbox_inches="tight")
plt.close()

# --- 3. Summary Statistics for Key Predictors ---
print("Computing descriptive health statistics...")
summary_stats = df.describe().T[["mean", "50%", "std"]].rename(columns={"50%": "median"})
summary_dict = summary_stats.round(3).to_dict(orient="index")

# Load existing metrics JSON if available
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as fh:
        try:
            metrics_data = json.load(fh)
        except Exception:
            metrics_data = {}
else:
    metrics_data = {}

metrics_data["Feature Correlations"] = top_corr.to_dict()
metrics_data["Health Summary Stats"] = summary_dict

with open(metrics_path, "w") as fh:
    json.dump(metrics_data, fh, indent=4)

print("✅ Insight PNG + EPS graphs generated successfully in /assets/")
