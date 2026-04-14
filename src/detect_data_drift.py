import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from alibi_detect.cd import TabularDrift
import os

print("1. Loading Data and Recreating Splits...")
df = pd.read_csv('../data/creditcard.csv')
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

# Recreate the exact same splits
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n2. Recreating the Drifted Data...")
X_test_drifted = X_test.copy()
noise_scale = 2.5 
np.random.seed(42)
# We are maliciously altering V1-V4 just like in the simulate_drift.py script
X_test_drifted['V1'] += np.random.normal(0, noise_scale, size=len(X_test_drifted))
X_test_drifted['V2'] += np.random.normal(0, noise_scale, size=len(X_test_drifted))
X_test_drifted['V3'] -= np.random.normal(0, noise_scale, size=len(X_test_drifted))
X_test_drifted['V4'] += np.random.normal(0, noise_scale, size=len(X_test_drifted))

print("\n3. Initializing Alibi-Detect (Tabular Drift)...")
# TabularDrift uses the Kolmogorov-Smirnov (K-S) test for continuous numerical features.
# p_val=0.05 means we want 95% confidence before flagging a feature as "drifted".
cd = TabularDrift(X_train.values, p_val=0.05)

print("\n4. Running Drift Detection on HEALTHY Test Data (Control Group)...")
preds_healthy = cd.predict(X_test.values)
is_drift_healthy = preds_healthy['data']['is_drift']
print(f"Overall Data Drift Detected in Healthy Data? {'🚨 YES' if is_drift_healthy == 1 else '✅ NO'}")

print("\n5. Running Drift Detection on DRIFTED Test Data (Experimental Group)...")
preds_drifted = cd.predict(X_test_drifted.values)
is_drift_experimental = preds_drifted['data']['is_drift']
print(f"Overall Data Drift Detected in Drifted Data? {'🚨 YES' if is_drift_experimental == 1 else '✅ NO'}")

print("\n📊 Feature-by-Feature Breakdown (Drifted Data):")
feature_names = X.columns
drifted_features = []

# Extract which specific features Alibi-Detect caught using the p-values
for i, feature in enumerate(feature_names):
    p_val = preds_drifted['data']['p_val'][i]
    # If the p-value is less than our 0.05 threshold, the feature has drifted
    if p_val < 0.05:
        drifted_features.append(f"{feature} (p-value: {p_val:.4e})")

print("Alibi-Detect flagged the following features as shifted:")
for feat in drifted_features:
    print(f" - {feat}")

print("\n✅ Alibi-Detect Baseline Complete.")