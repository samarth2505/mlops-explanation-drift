import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import wasserstein_distance
import pickle

print("1. Loading Baseline Data and Model...")
df = pd.read_csv('../data/creditcard.csv')
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_sample_indices = X_test.sample(2000, random_state=42).index

with open('../models/baseline_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

print("2. Establishing Absolute Baseline SHAP...")
explainer = shap.TreeExplainer(model)
shap_base = explainer.shap_values(X_test.loc[X_test_sample_indices])
if isinstance(shap_base, list):
    shap_base = shap_base[1]

v4_index = list(X.columns).index('V4')

print("\n3. Running Concept Shift Simulation (10 Stages)...")
print("Notice: The data (X) remains clean. Only the ground truth (Y) changes.")
print(f"{'Stage':<10} | {'Flipped %':<10} | {'F1-Score':<10} | {'EDM (V4)':<10}")
print("-" * 50)

results = []

for stage in range(11):
    flip_percentage = stage * 0.05  # Flip 0% to 50% of the labels
    
    # Simulate Concept Shift: The world changes the definition of Fraud
    y_stage_concept = y_test.copy()
    
    # We randomly select a percentage of normal transactions and declare them "Fraud"
    # This simulates a new type of fraud emerging that looks exactly like normal behavior
    np.random.seed(42 + stage) 
    normal_indices = y_stage_concept[y_stage_concept == 0].index
    num_to_flip = int(len(normal_indices) * flip_percentage)
    
    if num_to_flip > 0:
        flip_idx = np.random.choice(normal_indices, num_to_flip, replace=False)
        y_stage_concept.loc[flip_idx] = 1
    
    # 1. Measure Model Performance (F1 will drop because the 'answer key' changed)
    preds = model.predict(X_test)
    current_f1 = f1_score(y_stage_concept, preds)
    
    # 2. Extract New Explanations (X hasn't changed, so SHAP shouldn't change!)
    shap_current = explainer.shap_values(X_test.loc[X_test_sample_indices])
    if isinstance(shap_current, list):
        shap_current = shap_current[1]
        
    # 3. Calculate EDM
    current_edm = wasserstein_distance(shap_base[:, v4_index], shap_current[:, v4_index])
    
    results.append({
        'Stage': stage,
        'Flip_Percentage': flip_percentage,
        'F1_Score': current_f1,
        'EDM_Score': current_edm
    })
    
    print(f"Stage {stage:<4} | {flip_percentage:<10.2f} | {current_f1:<10.4f} | {current_edm:<10.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv('../data/concept_shift_results.csv', index=False)
print("\n✅ Concept Shift experiment complete. Data saved for plotting.")