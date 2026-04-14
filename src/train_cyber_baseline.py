import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import os
from sklearn.model_selection import train_test_split

print("1. Loading and Cleaning Cybersecurity Data...")
try:
    df = pd.read_csv('../data/cicids_ddos.csv')
except FileNotFoundError:
    print("❌ Error: Ensure 'cicids_ddos.csv' is in your 'data/' folder.")
    exit()

# Clean invisible spaces and broken numbers
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Map target to binary
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Sample to 100k rows to keep processing times reasonable on your Mac
df = df.sample(100000, random_state=42)

X = df.drop(columns=['Label'])
y = df['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("2. Training the Cybersecurity Baseline Model...")
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

print("3. Saving the Frozen Model...")
os.makedirs('../models', exist_ok=True)
with open('../models/cyber_xgboost.pkl', 'wb') as f:
    pickle.dump(model, f)

print("4. Extracting and Saving Baseline SHAP Values...")
# We save a specific sample of 1,000 test rows to use consistently across all attacks
X_test_sample = X_test.sample(1000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_base = explainer.shap_values(X_test_sample)

if isinstance(shap_base, list):
    shap_base = shap_base[1]

# Save the matrix and the exact test data used
np.save('../data/cyber_baseline_shap.npy', shap_base)
X_test_sample.to_csv('../data/cyber_X_test_sample.csv', index=False)
y_test.loc[X_test_sample.index].to_csv('../data/cyber_y_test_sample.csv', index=False)

print("✅ Foundation built! Model and baseline explanations saved to disk.")