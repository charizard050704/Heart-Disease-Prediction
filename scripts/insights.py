# scripts/insights.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- Uniform plot settings (professor requirements) ---
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Ensure directories
os.makedirs("assets", exist_ok=True)
os.makedirs("data", exist_ok=True)

data_path = "data/heart.csv"
metrics_path = "assets/model_metrics_extended.json"

print("Loading dataset for correlation and insights...")
df = pd.read_csv(data_path)

# --- 1. Correlation Heatmap ---
print("Generating correlation heatmap...")
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
# plt.title removed as per professor requirement
plt.tight_layout()
plt.savefig("assets/correlation_heatmap.eps", format='eps', bbox_inches="tight")
plt.close()

# --- 2. Top Features Driving Heart Risk ---
print("Computing top correlated features with 'target'...")
if "target" in df.columns:
    target_corr = corr["target"].drop("target").sort_values(key=lambda s: np.abs(s), ascending=False)
    top_corr = target_corr.head(10)
else:
    top_corr = pd.Series(dtype=float)

plt.figure(figsize=(7,5))
sns.barplot(x=top_corr.values, y=top_corr.index, hue=top_corr.index, palette="Reds_r", legend=False)
plt.ylabel("") 
# plt.title removed
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

print("✅ Correlation heatmap, top-risk bar chart, and summary stats saved in EPS format in /assets/")
