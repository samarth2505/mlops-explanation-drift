import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle

print("1. Loading the Frozen Model and Test Data...")
try:
    with open('../models/cyber_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    X_test = pd.read_csv('../data/cyber_X_test_sample.csv')
    y_test = pd.read_csv('../data/cyber_y_test_sample.csv')['Label']
except FileNotFoundError:
    print("❌ Error: Missing foundation files.")
    exit()

print("2. Creating the 100% Mimicry Disguised Data...")
# We only want to look at the hackers (DDoS packets)
ddos_indices = y_test[y_test == 1].index
benign_indices = y_test[y_test == 0].index

target_features = ['Average Packet Size', 'Bwd Packet Length Mean', 'Flow Bytes/s']
benign_profiles = X_test.loc[benign_indices, target_features].mean()

# Apply the perfect disguise to the hackers
X_disguised = X_test.copy()
for feat in target_features:
    X_disguised.loc[ddos_indices, feat] = benign_profiles[feat]

print("3. Ripping Open the Math with SHAP...")
explainer = shap.TreeExplainer(model)

# We ask SHAP: "How did you evaluate ONLY the disguised hackers?"
X_ddos_disguised = X_disguised.loc[ddos_indices]
shap_disguised = explainer.shap_values(X_ddos_disguised)

if isinstance(shap_disguised, list):
    shap_disguised = shap_disguised[1]

# Calculate the average "Vote" each feature cast for these disguised packets
# Positive = Voted DDoS | Negative = Voted Benign
mean_shap_votes = shap_disguised.mean(axis=0)

df_proof = pd.DataFrame({
    'Feature': X_test.columns,
    'DDoS_Vote_Strength': mean_shap_votes
})

print("\n🕵️‍♂️ THE VERDICT: How the Model Caught the Disguise 🕵️‍♂️")
print("-" * 65)
print("Did the disguise work on our targeted features?")
for feat in target_features:
    vote = df_proof[df_proof['Feature'] == feat]['DDoS_Vote_Strength'].values[0]
    print(f"{feat:25} : {vote:>8.4f} (If negative/near 0, the disguise fooled this feature)")

print("\n🚨 So what actually triggered the alarm? (The Top 5 Undisguised Features):")
top_catchers = df_proof.sort_values(by='DDoS_Vote_Strength', ascending=False).head(5)
print(top_catchers.to_string(index=False))

print("-" * 65)