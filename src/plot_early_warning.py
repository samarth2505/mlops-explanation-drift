import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the plots directory exists
os.makedirs('../plots', exist_ok=True)

print("1. Loading Progressive Drift Data...")
try:
    df = pd.read_csv('../data/progressive_drift_results.csv')
except FileNotFoundError:
    print("❌ Error: Could not find data. Run experiment_early_warning.py first.")
    exit()

print("2. Generating The Early Warning Plot...")
# Set up the visual style
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot 1: F1-Score (The Traditional Metric) on the Left Y-Axis
color1 = 'tab:blue'
ax1.set_xlabel('Drift Severity (Noise Level)', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1-Score (Model Accuracy)', color=color1, fontsize=12, fontweight='bold')
line1 = ax1.plot(df['Noise_Level'], df['F1_Score'], color=color1, marker='o', linewidth=2, label='F1-Score (Traditional)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.5, 1.0) # Keep F1 scale logical

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  

# Plot 2: EDM Score (Your Novel Metric) on the Right Y-Axis
color2 = 'tab:red'
ax2.set_ylabel('Explanation Drift Metric (EDM)', color=color2, fontsize=12, fontweight='bold')
line2 = ax2.plot(df['Noise_Level'], df['EDM_Score'], color=color2, marker='s', linewidth=2, linestyle='--', label='EDM (Novel Signal)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, df['EDM_Score'].max() + 0.2)

# Title and Layout
plt.title('Early Warning Detection: EDM vs. F1-Score under Covariate Shift', fontsize=14, fontweight='bold', pad=15)

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', frameon=True, fontsize=11)

plt.tight_layout()
save_path = '../plots/early_warning_signal.png'
plt.savefig(save_path, dpi=300)
print(f"✅ Success! Plot saved to {save_path}")