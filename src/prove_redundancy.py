import numpy as np
import pandas as pd
import os

print("1. Loading SHAP data from the Cybersecurity Experiment...")
try:
    df = pd.read_csv('../data/cicids_ddos.csv')
    df.columns = df.columns.str.strip()
    feature_names = df.drop(columns=['Label']).columns.tolist()
    
    # We load the baseline and the drifted SHAP values we saved (you'll need to run experiment_cybersecurity.py 
    # and save the Stage 10 SHAP values, or we can just extract them dynamically. Let's do it cleanly).
except FileNotFoundError:
    print("❌ Error: Missing dataset.")
    exit()

# Let's recreate the logic quickly just to extract the raw importance arrays
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

# Clean and split
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
df = df.sample(100000, random_state=42)
X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_sample = X_test.sample(1000, random_state=42)

print("2. Re-running the healthy vs attacked models for extraction...")
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)

# Healthy SHAP
shap_base = explainer.shap_values(X_test_sample)
if isinstance(shap_base, list): shap_base = shap_base[1]

# Poisoned SHAP (Stage 10 from previous experiment)
X_poisoned = X_test_sample.copy()
target_features = ['Average Packet Size', 'Bwd Packet Length Mean', 'Max Packet Length', 'Flow Bytes/s']
for feat in target_features:
    X_poisoned[feat] += np.random.normal(0, 5000.0, size=len(X_poisoned))

shap_poisoned = explainer.shap_values(X_poisoned)
if isinstance(shap_poisoned, list): shap_poisoned = shap_poisoned[1]

print("\n3. Calculating Global Feature Importance (Mean Absolute SHAP)...")
# Calculate the average impact of each feature
importance_base = np.abs(shap_base).mean(axis=0)
importance_poisoned = np.abs(shap_poisoned).mean(axis=0)

df_imp = pd.DataFrame({
    'Feature': feature_names,
    'Healthy_Importance': importance_base,
    'Poisoned_Importance': importance_poisoned
})

# Calculate the difference to see what "stepped up" to save the model
df_imp['Shift'] = df_imp['Poisoned_Importance'] - df_imp['Healthy_Importance']

print("\n🏆 Top 5 Features the Model LOST Confidence in (Poisoned):")
print(df_imp.sort_values(by='Shift', ascending=True).head(5)[['Feature', 'Shift']].to_string(index=False))

print("\n🛡️ Top 5 Features that STEPPED UP to save the accuracy (Redundancy):")
print(df_imp.sort_values(by='Shift', ascending=False).head(5)[['Feature', 'Shift']].to_string(index=False))