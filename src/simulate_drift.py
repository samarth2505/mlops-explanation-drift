import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print("1. Loading Data and Recreating Test Set...")
df = pd.read_csv('../data/creditcard.csv')
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

# Recreate the exact same test set from the baseline
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n2. Injecting Data Drift (Simulating New Fraud Tactics)...")
X_test_drifted = X_test.copy()

# We simulate drift by adding severe noise to some of the highest-variance PCA features (V1, V2, V3, V4)
# This represents a shift in user/fraudster behavior profiles
noise_scale = 2.5 
np.random.seed(42)
X_test_drifted['V1'] += np.random.normal(0, noise_scale, size=len(X_test_drifted))
X_test_drifted['V2'] += np.random.normal(0, noise_scale, size=len(X_test_drifted))
X_test_drifted['V3'] -= np.random.normal(0, noise_scale, size=len(X_test_drifted))
X_test_drifted['V4'] += np.random.normal(0, noise_scale, size=len(X_test_drifted))

print("3. Loading Baseline Model and Predicting on Drifted Data...")
with open('../models/baseline_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

preds_drifted = model.predict(X_test_drifted)
print(f"Drifted Accuracy: {accuracy_score(y_test, preds_drifted):.4f}")
print("Drifted Classification Report:\n", classification_report(y_test, preds_drifted))

print("\n4. Generating Drifted SHAP Explanations...")
# We use the same 2000 sample indices for a fair 1-to-1 comparison
X_test_drifted_sample = X_test_drifted.sample(2000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values_drifted = explainer.shap_values(X_test_drifted_sample)

np.save('../data/drifted_shap_values.npy', shap_values_drifted)
print("✅ Drifted SHAP values saved to ../data/drifted_shap_values.npy")