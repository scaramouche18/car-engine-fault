# Car Engine Fault Detection

A machine learning project that classifies engine condition (Normal, Minor Fault, Critical Fault) using sensor data — vibration, acoustic, temperature, and pressure readings.

## Problem Statement

Early detection of engine faults can prevent costly breakdowns and improve safety. This project explores whether sensor telemetry data can be used to classify engine health into three states using supervised machine learning.

## Dataset

- **Source:** Kaggle — Engine Fault Detection Data
- **Size:** 10,000 rows, 11 sensor features, 1 target column
- **Features:** Vibration Amplitude, RMS Vibration, Vibration Frequency, Surface Temperature, Exhaust Temperature, Acoustic dB, Acoustic Frequency, Intake Pressure, Exhaust Pressure, Frequency Band Energy, Amplitude Mean
- **Target:** Engine_Condition — 0 (Normal), 1 (Minor Fault), 2 (Critical Fault)
- **Class distribution:** Normal 59.6%, Minor Fault 30.4%, Critical Fault 10.0% — moderately imbalanced

## Approach

1. **EDA:** Boxplots and correlation analysis showed no single sensor feature linearly separates the three engine conditions — ruling out simple threshold-based rules and motivating a multivariate ML approach.
2. **Baseline model:** Logistic Regression, to establish a reference point.
3. **Tree-based models:** Random Forest and XGBoost, to capture non-linear, multivariate patterns.
4. **Imbalance handling:** Compared class_weight='balanced' against SMOTE (synthetic oversampling of the minority class).
5. **Evaluation:** Used precision, recall, and F1-score per class (not just accuracy) due to class imbalance.

## Results

| Model | Accuracy | Critical Fault Recall |
|---|---|---|
| Logistic Regression | 32% | 0.41 |
| Random Forest (default) | 60% | 0.00 |
| Random Forest (tuned) | 54% | 0.04 |
| XGBoost | 55% | 0.00 |
| Random Forest + SMOTE | 49% | 0.13 |

**Key finding:** Higher-accuracy models (Random Forest, XGBoost) almost completely failed to detect the rare Critical Fault class, despite class weighting. SMOTE improved Critical Fault recall at the cost of overall accuracy — a deliberate tradeoff, since missing a real fault is more costly than a false alarm.

The confusion matrix showed the model's biggest weakness was misclassifying both Minor and Critical faults as Normal — the highest-risk type of error for a real-world diagnostic system.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), Matplotlib, Seaborn, Streamlit

## Project Structure

- data/ — Dataset and generated plots
- notebooks/ — EDA and model experimentation
- models/ — Saved trained model
- src/ — Scripts
- README.md

## Limitations & Future Work

- Dataset appears synthetically generated (clean, bounded ranges across all features) rather than real-world telemetry.
- Class imbalance (10% Critical Fault) limits model performance on the most safety-critical class.
- Future improvements: collect more Critical Fault examples, try threshold tuning per class, explore additional feature engineering.

## How to Run

Install dependencies: pip install -r requirements.txt

Then open the notebook: jupyter notebook notebooks/01_eda.ipynb
