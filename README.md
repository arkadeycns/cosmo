# Customer Next Month Spend Prediction

Predicting how much a customer will spend next month (`spend_next_month`) from their demographics and past shopping behaviour.

## Notebooks

Run them in order. Each one saves its output for the next.

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_preprocessing.ipynb` | load, inspect and clean the raw data |
| 02 | `02_eda.ipynb` | exploratory analysis and key insights |
| 03 | `03_feature_engineering.ipynb` | redundancy, skew, outlier and bias checks, new features, encoding comparison |
| 04 | `04_model_training.ipynb` | train/test split, Linear Regression, Random Forest, XGBoost, ANN |
| 05 | `05_tuning.ipynb` | hyperparameter tuning with Optuna |
| 06 | `06_validation.ipynb` | test set comparison, residuals, feature importance |

## Results (test set)

| Model | MAE | R2 |
|---|---|---|
| **XGBoost (Optuna)** | **102.1** | **0.837** |
| Random Forest (Optuna) | 103.2 | 0.834 |
| ANN (Optuna) | 104.2 | 0.836 |
| Linear Regression (baseline) | 116.5 | 0.786 |

## App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Predict for a single customer, or upload a CSV (try `sample_test.csv`) and download the predictions.

> The Random Forest model files are not in the repo (too large for GitHub). Run notebooks 04 and 05 to recreate them.

## Team

| Member | Part |
|---|---|
| Ashish | preprocessing, EDA |
| Arka | feature engineering, app |
| Shivansh | model training, tuning |
| Ritesh | validation |
