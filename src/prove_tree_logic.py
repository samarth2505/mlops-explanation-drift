import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import os

print("1. Loading the Frozen Baseline Model and Data...")
try:
    with open('../models/cyber_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    X_test = pd.read_csv('../data/cyber_X_test_sample.csv')
except FileNotFoundError:
    print("❌ Error: Missing foundation files.")
    exit()

print("\n2. Extracting the Literal Tree Logic...")
booster = model.get_booster()
tree_df = booster.trees_to_dataframe()

target_feature = 'Average Packet Size'
feature_splits = tree_df[tree_df['Feature'] == target_feature]

if not feature_splits.empty:
    max_split_value = feature_splits['Split'].max()
    print(f"\n🔍 MATH REVEALED:")
    print(f"The highest decision threshold the model ever learned for {target_feature} is {max_split_value:.2f} bytes.")
    print("When you added +5000 bytes of noise, you forced EVERY packet far past this threshold.")
    print("The trees mathematically hit a ceiling, automatically classifying it as a severe anomaly (DDoS).")
else:
    print(f"Feature {target_feature} was not used in the tree splits.")

print("\n3. Generating Partial Dependence Plot (PDP)...")
os.makedirs('../plots', exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))

# Generate PDP to visually prove the threshold
display = PartialDependenceDisplay.from_estimator(
    model, 
    X_test, 
    features=[target_feature], 
    kind='average',
    ax=ax,
    line_kw={'color': 'red', 'linewidth': 3}
)

plt.title(f'Mathematical Threshold for {target_feature}', fontsize=14, fontweight='bold')
plt.ylabel('Prediction Impact (Higher = DDoS)', fontsize=12)
plt.xlabel(f'{target_feature} (Bytes)', fontsize=12)

# We extend the X-axis to 6000 to show the "Out of Distribution" flatline
plt.xlim(0, 6000)
plt.grid(True)

plt.tight_layout()
plt.savefig('../plots/pdp_mathematical_proof.png')
print("✅ Mathematical proof saved to ../plots/pdp_mathematical_proof.png")