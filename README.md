# Explanation Drift Metric (EDM) for MLOps

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-success.svg)

## Overview
Traditional MLOps monitoring relies heavily on accuracy metrics (F1-Score, Precision) or raw data distribution checks (Covariate Shift). However, in highly redundant or over-parameterized environments, a model's internal reasoning can fundamentally degrade or be hijacked by an attacker while its accuracy remains artificially high. 

This repository introduces and implements the **Explanation Drift Metric (EDM)**. By calculating the Wasserstein Distance (Earth Mover's Distance) between baseline SHAP values and production SHAP values, EDM acts as a highly sensitive, early-warning "tripwire" for model decay and adversarial evasion attacks.

## Core Experiments & Findings

### 1. The Early Warning System (Financial Sector)
* **Dataset:** Imbalanced Credit Card Fraud (Kaggle).
* **Experiment:** Simulated progressive Covariate Shift via Gaussian noise injection.
* **Result:** EDM successfully detected the model's logic breaking down several stages before traditional F1-Scores showed significant degradation.

### 2. The Concept Shift Paradox
* **Experiment:** Maintained clean input distributions while maliciously flipping ground-truth labels to simulate a shifting real-world environment.
* **Result:** EDM remained at 0.0, successfully proving its utility as a diagnostic tool that isolates pure model-logic failures from external ground-truth shifts.

### 3. Multi-Vector Evasion & Mimicry (Cybersecurity)
* **Dataset:** CICIDS2017 (Network Intrusion / DDoS).
* **Experiment:** Executed a Surgical Mimicry Attack, forcing DDoS packets to perfectly mimic Benign traffic profiles across critical features.
* **Result:** Proved mathematically (via Partial Dependence Plots and SHAP vote extraction) how highly dimensional models (78+ features) use feature redundancy to survive targeted attacks, and how EDM captures this "attribution shift."

## Repository Structure
```text
explanation-drift/
│
├── data/                  # Ignored: Raw CSVs and extracted SHAP matrices (.npy)
├── models/                # Ignored: Frozen XGBoost models (.pkl)
├── plots/                 # Generated visualizations (EDM vs F1, PDPs, SHAP Summaries)
├── src/                   # Core pipeline and experimental scripts
│   ├── train_*.py         # Baseline training scripts
│   ├── experiment_*.py    # Covariate and Concept shift simulations
│   ├── attack_*.py        # Adversarial mimicry and noise attacks
│   ├── prove_*.py         # Mathematical extraction (PDPs, Tree Logic, SHAP Votes)
│   └── plot_*.py          # Visualization generators
│
├── requirements.txt       # Project dependencies
└── README.md