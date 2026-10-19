# Predictive Maintenance for Turbofan Engines

Predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines from sensor data and raises a maintenance alert when failure is near.

**Live demo:** https://predictive-maintenance-dikshana.streamlit.app

## Problem
Unplanned engine failures are costly. If we can predict how many cycles an engine has left, maintenance can be scheduled before it fails.

## Dataset
NASA C-MAPSS turbofan degradation dataset, subset FD001 (100 engines, 21 sensors, run-to-failure).

## Approach
1. Computed RUL for every cycle and capped it at 125.
2. Removed 7 constant sensors, kept 14.
3. Added 10-cycle rolling-mean features per engine to reduce noise.
4. Split train/validation by engine (80/20) to avoid leakage.
5. Trained Random Forest and XGBoost models.
6. Explained the model with SHAP.
7. Tuned the alert trigger for precision vs recall.
8. Built a Streamlit dashboard.

## Results (held-out engines)

| Model | RMSE (capped RUL) |
|---|---|
| Random Forest, raw sensors (baseline) | 19.09 |
| Random Forest, rolling features + capped RUL | 17.68 |
| XGBoost, rolling features + capped RUL | **17.39** |

Failure alert (failure = RUL <= 30), XGBoost:

| Alert when predicted RUL <= | Precision | Recall |
|---|---|---|
| 30 | 0.913 | 0.829 |
| 35 | 0.829 | 0.894 |
| 40 | 0.746 | 0.932 |
| 45 | 0.669 | 0.953 |
| 50 | 0.617 | 0.973 |

Raising the trigger catches more failures (higher recall) but causes more false alarms (lower precision).

## Explainability
SHAP showed that rolling-averaged sensors s4, s3, s2 and s9 had the largest impact on predicted RUL.

## Limitations
- The dashboard demo includes engines used in training, so its curves look better than real performance. The RMSE above is from held-out engines.
- Only FD001 (single operating condition) is used.
- No LSTM or deep learning model.

## Tech stack
Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
