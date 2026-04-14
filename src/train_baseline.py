import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# 1. Setup paths
DATA_PATH = '../data/creditcard.csv'
MODEL_DIR = '../models'
os.makedirs(MODEL_DIR, exist_ok=True)

print("1. Loading Kaggle Credit Card Fraud Dataset...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"❌ Error: Could not find dataset at {DATA_PATH}. Please download it from Kaggle.")
    exit()

# The 'Time' feature is often dropped for pure behavior analysis, but we'll keep 'Amount' and 'V1-V28'
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

# 2. Split into Baseline (Train/Test)
# We stratify to ensure the 0.17% fraud rate is maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("2. Training Baseline XGBoost Model...")
# scale_pos_weight helps XGBoost handle the extreme imbalance in fraud data
model = xgb.XGBClassifier(
    eval_metric='logloss', 
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate to prove it's a "good" model
preds = model.predict(X_test)
print(f"Baseline Accuracy: {accuracy_score(y_test, preds):.4f}")
print("Classification Report:\n", classification_report(y_test, preds))

# Save the model
model_path = os.path.join(MODEL_DIR, 'baseline_xgboost.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

print("3. Generating Baseline SHAP Explanations (This might take a minute)...")
# We use a sample of the test set for SHAP to speed up computation locally
X_test_sample = X_test.sample(2000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)

# Save baseline SHAP values for EDM comparison
shap_path = '../data/baseline_shap_values.npy'
np.save(shap_path, shap_values)
print(f"✅ Baseline SHAP values saved to {shap_path}")