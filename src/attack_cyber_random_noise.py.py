import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
from sklearn.metrics import f1_score
from scipy.stats import wasserstein_distance

print("1. Loading Foundation Data and Model...")
try:
    with open('../models/cyber_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    shap_base = np.load('../data/cyber_baseline_shap.npy')
    X_test = pd.read_csv('../data/cyber_X_test_sample.csv')
    y_test = pd.read_csv('../data/cyber_y_test_sample.csv')['Label']
except FileNotFoundError:
    print("❌ Error: Missing foundation files. Run train_cyber_baseline.py first.")
    exit()

explainer = shap.TreeExplainer(model)

# We attack the cluster of features most critical to DDoS detection
target_features = ['Average Packet Size', 'Bwd Packet Length Mean', 'Max Packet Length', 'Flow Bytes/s']
track_feat_idx = list(X_test.columns).index('Average Packet Size')

print("\n2. Running Random Noise Attack Simulation (10 Stages)...")
print(f"{'Stage':<10} | {'Noise LvL':<10} | {'F1-Score':<10} | {'EDM (Avg Pkt Size)':<10}")
print("-" * 65)

results = []

for stage in range(11):
    # Crank the noise multiplier way up
    noise_scale = stage * 500.0  
    
    X_stage = X_test.copy()
    np.random.seed(42)
    
    # The attacker maliciously obscures multiple packet characteristics by adding extreme noise
    for feat in target_features:
        X_stage[feat] += np.random.normal(0, noise_scale, size=len(X_stage))
    
    # 1. Measure Model Performance
    preds = model.predict(X_stage)
    current_f1 = f1_score(y_test, preds)
    
    # 2. Extract New Explanations
    shap_current = explainer.shap_values(X_stage)
    if isinstance(shap_current, list):
        shap_current = shap_current[1]
        
    # 3. Calculate EDM
    current_edm = wasserstein_distance(shap_base[:, track_feat_idx], shap_current[:, track_feat_idx])
    
    results.append({
        'Stage': stage,
        'Noise_Level': noise_scale,
        'F1_Score': current_f1,
        'EDM_Score': current_edm
    })
    
    print(f"Stage {stage:<4} | {noise_scale:<10.1f} | {current_f1:<10.4f} | {current_edm:<10.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv('../data/cyber_noise_results.csv', index=False)
print("\n✅ Random Noise Attack complete. Data saved.")