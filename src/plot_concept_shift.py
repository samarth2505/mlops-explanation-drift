import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("1. Loading Concept Shift Data...")
try:
    df = pd.read_csv('../data/concept_shift_results.csv')
except FileNotFoundError:
    print("❌ Error: Could not find data. Run experiment_concept_shift.py first.")
    exit()

print("2. Generating The Concept Shift Paradox Plot...")
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot 1: F1-Score crashing on the Left Y-Axis
color1 = 'tab:blue'
ax1.set_xlabel('Concept Shift Severity (% of Normal Labels Flipped to Fraud)', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1-Score (Model Accuracy)', color=color1, fontsize=12, fontweight='bold')
# Multiply by 100 to make the X-axis read as percentages (0 to 50)
line1 = ax1.plot(df['Flip_Percentage'] * 100, df['F1_Score'], color=color1, marker='o', linewidth=2, label='F1-Score (Traditional)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(-0.05, 1.0)

ax2 = ax1.twinx()  

# Plot 2: EDM Score staying flat on the Right Y-Axis
color2 = 'tab:red'
ax2.set_ylabel('Explanation Drift Metric (EDM)', color=color2, fontsize=12, fontweight='bold')
line2 = ax2.plot(df['Flip_Percentage'] * 100, df['EDM_Score'], color=color2, marker='s', linewidth=2, linestyle='--', label='EDM (Novel Signal)')
ax2.tick_params(axis='y', labelcolor=color2)
# We keep the Y-axis scale similar to the first plot so they look good side-by-side in your paper
ax2.set_ylim(-0.1, 1.2) 

plt.title('The Concept Shift Paradox: EDM Remains Flat as Accuracy Collapses', fontsize=14, fontweight='bold', pad=15)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, fontsize=11)

plt.tight_layout()
save_path = '../plots/concept_shift_paradox.png'
plt.savefig(save_path, dpi=300)
print(f"✅ Success! Plot saved to {save_path}")