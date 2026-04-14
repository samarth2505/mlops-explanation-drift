import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
from sklearn.metrics import f1_score
from scipy.stats import wasserstein_distance
import os

print("1. Loading Foundation Data and Model...")
with open('../models/cyber_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

shap_base = np.load('../data/cyber_baseline_shap.npy')
X_test = pd.read_csv('../data/cyber_X_test_sample.csv')
y_test = pd.read_csv('../data/cyber_y_test_sample.csv')['Label']

explainer = shap.TreeExplainer(model)

print("2. Calculating the 'Benign' Disguise...")
# Identify which rows are normal traffic and which are DDoS
benign_indices = y_test[y_test == 0].index
ddos_indices = y_test[y_test == 1].index

# We target the features that usually give away a DDoS attack
target_features = ['Average Packet Size', 'Bwd Packet Length Mean', 'Flow Bytes/s']

# Find exactly what the firewall considers "Normal" for these features
benign_profiles = X_test.loc[benign_indices, target_features].mean()

print("\nTarget Benign Profile for Disguise:")
print(benign_profiles.to_string())

track_feat_idx = list(X_test.columns).index('Average Packet Size')

print("\n3. Running Surgical Mimicry Attack (10 Stages)...")
print(f"{'Stage':<10} | {'Mimicry %':<10} | {'F1-Score':<10} | {'EDM (Avg Pkt Size)':<10}")
print("-" * 65)

results = []

for stage in range(11):
    # Mimicry goes from 0% (obvious DDoS) to 100% (perfectly mimics normal traffic)
    mimicry_factor = stage * 0.10  
    
    X_stage = X_test.copy()
    
    # Forcefully morph the DDoS packets to look like Benign packets
    for feat in target_features:
        original_ddos_values = X_stage.loc[ddos_indices, feat]
        target_benign_value = benign_profiles[feat]
        
        # Shift the value toward the benign mean based on the mimicry factor
        morphed_values = original_ddos_values * (1 - mimicry_factor) + (target_benign_value * mimicry_factor)
        X_stage.loc[ddos_indices, feat] = morphed_values
        
    # 1. Measure Model Performance (Will the firewall get fooled?)
    preds = model.predict(X_stage)
    current_f1 = f1_score(y_test, preds)
    
    # 2. Extract Explanations
    shap_current = explainer.shap_values(X_stage)
    if isinstance(shap_current, list):
        shap_current = shap_current[1]
        
    # 3. Calculate EDM
    current_edm = wasserstein_distance(shap_base[:, track_feat_idx], shap_current[:, track_feat_idx])
    
    results.append({
        'Stage': stage,
        'Mimicry_Factor': mimicry_factor * 100,
        'F1_Score': current_f1,
        'EDM_Score': current_edm
    })
    
    print(f"Stage {stage:<4} | {mimicry_factor*100:<10.0f} | {current_f1:<10.4f} | {current_edm:<10.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv('../data/cyber_mimicry_results.csv', index=False)
print("\n✅ Mimicry Attack complete. Data saved for plotting.")