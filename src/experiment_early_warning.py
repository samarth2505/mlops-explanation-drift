import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import wasserstein_distance
import pickle
import os

print("1. Loading Baseline Data and Model...")
df = pd.read_csv('../data/creditcard.csv')
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

# We need the FULL test set for an accurate F1 Score
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# We grab the 2000 indices so we can use them for SHAP later (for speed)
X_test_sample_indices = X_test.sample(2000, random_state=42).index

with open('../models/baseline_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

print("2. Establishing Absolute Baseline SHAP...")
explainer = shap.TreeExplainer(model)
# Calculate baseline SHAP using the 2000 sample
shap_base = explainer.shap_values(X_test.loc[X_test_sample_indices])
if isinstance(shap_base, list):
    shap_base = shap_base[1]

v4_index = list(X.columns).index('V4')

print("\n3. Running Progressive Drift Simulation (10 Stages)...")
print(f"{'Stage':<10} | {'Noise LvL':<10} | {'F1-Score':<10} | {'EDM (V4)':<10}")
print("-" * 50)

results = []

for stage in range(11):
    # We hit it with a heavier noise multiplier to force the trees to break
    noise_scale = stage * 1.5  
    
    # Drift the ENTIRE test set
    X_stage_full = X_test.copy()
    np.random.seed(42)
    X_stage_full['V4'] += np.random.normal(0, noise_scale, size=len(X_stage_full))
    X_stage_full['V2'] += np.random.normal(0, noise_scale, size=len(X_stage_full))
    
    # 1. Measure Model Performance on the FULL 56k+ rows
    preds = model.predict(X_stage_full)
    current_f1 = f1_score(y_test, preds)
    
    # 2. Extract New Explanations (Only on the 2000 sample for speed)
    X_stage_sample = X_stage_full.loc[X_test_sample_indices]
    shap_current = explainer.shap_values(X_stage_sample)
    if isinstance(shap_current, list):
        shap_current = shap_current[1]
        
    # 3. Calculate EDM
    current_edm = wasserstein_distance(shap_base[:, v4_index], shap_current[:, v4_index])
    
    results.append({
        'Stage': stage,
        'Noise_Level': noise_scale,
        'F1_Score': current_f1,
        'EDM_Score': current_edm
    })
    
    print(f"Stage {stage:<4} | {noise_scale:<10.1f} | {current_f1:<10.4f} | {current_edm:<10.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv('../data/progressive_drift_results.csv', index=False)
print("\n✅ Progressive experiment complete. Data saved for plotting.")