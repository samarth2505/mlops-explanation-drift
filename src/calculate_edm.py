import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine

print("1. Loading SHAP Matrices...")
try:
    shap_base = np.load('../data/baseline_shap_values.npy')
    shap_drift = np.load('../data/drifted_shap_values.npy')
except FileNotFoundError:
    print("❌ Error: Run train_baseline.py and simulate_drift.py first.")
    exit()

# SHAP returns a list if it's a multi-class/binary setup depending on the XGBoost version.
# We ensure we are grabbing the matrix for the positive class (Fraud)
if isinstance(shap_base, list):
    shap_base = shap_base[1]
    shap_drift = shap_drift[1]

# Reconstruct feature names (Amount + V1 to V28)
feature_names = [f"V{i}" for i in range(1, 29)] + ['Amount']

print("\n2. Calculating Explanation Drift Metric (EDM) per feature...")
drift_metrics = []

# Calculate statistical shift in explanation distributions for each feature
for i, feature in enumerate(feature_names):
    # Extract the array of SHAP values for this specific feature across all 2000 samples
    base_feature_shap = shap_base[:, i]
    drift_feature_shap = shap_drift[:, i]
    
    # Calculate Wasserstein Distance (Earth Mover's Distance)
    # This measures how much 'work' it takes to turn the baseline SHAP distribution into the drifted one
    w_dist = wasserstein_distance(base_feature_shap, drift_feature_shap)
    
    drift_metrics.append({
        'Feature': feature,
        'Wasserstein_EDM': w_dist
    })

# Convert to DataFrame and sort by the most heavily drifted explanations
df_edm = pd.DataFrame(drift_metrics)
df_edm = df_edm.sort_values(by='Wasserstein_EDM', ascending=False).reset_index(drop=True)

print("\n🏆 Top 10 Features with the Highest Explanation Drift:")
print(df_edm.head(10).to_string(index=False))

print("\n3. Calculating Global Explanation Shift...")
# Calculate the Mean Absolute SHAP (Global Feature Importance)
base_global_importance = np.mean(np.abs(shap_base), axis=0)
drift_global_importance = np.mean(np.abs(shap_drift), axis=0)

# Cosine Distance between the global importance vectors
# 0 means vectors point exactly the same way, 1 means orthogonal
global_cosine_drift = cosine(base_global_importance, drift_global_importance)

print(f"Global Cosine Explanation Drift: {global_cosine_drift:.4f}")
if global_cosine_drift > 0.05:
    print("⚠️ WARNING: Significant shift in model reasoning detected!")
else:
    print("✅ Model reasoning remains relatively stable.")