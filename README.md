# ML Churn Prediction Pipeline

An end-to-end machine learning pipeline for predicting customer churn. Built as a learning project covering synthetic data generation, feature engineering, model training with experiment tracking, a REST inference API, and data drift monitoring.

---

## What is Customer Churn?

Customer churn is when a customer stops using a product or service. Predicting it before it happens lets businesses take action — offering discounts, improving support, or targeting at-risk customers proactively. It's one of the most common real-world ML problems across telecoms, SaaS, banking, and retail.

---

## Project Structure

```
ml-churn-pipeline/
├── src/
│   ├── generate_data.py   — synthetic dataset generation
│   ├── train.py           — model training, evaluation, SHAP, MLflow tracking
│   └── monitor.py         — Evidently drift and classification reports
├── api/
│   └── main.py            — FastAPI inference server
├── data/                  — generated CSVs (gitignored)
├── models/                — saved model artifacts (gitignored)
├── reports/               — Evidently HTML reports (gitignored)
└── requirements.txt
```

---

## Current Progress

### Data Generation (`src/generate_data.py`)

Generates a synthetic dataset of 10,000 customers with realistic churn behaviour. Rather than assigning churn randomly, it uses a **logistic formula** to simulate real-world drivers:

```
logit = -3.5
      + 0.04 × (monthly_charges − 65)     ← high bills increase risk
      − 0.03 × tenure                      ← loyal customers stay longer
      + 0.50 × num_complaints              ← complaints are a strong signal
      + 0.02 × days_since_last_login       ← disengagement predicts leaving
      + 0.80 × (contract == month-to-month)← easiest contract to leave
      − 0.50 × (contract == two-year)      ← locked in, less likely to churn
      + 0.30 × (internet == fiber)         ← expensive service, more churn
      − 0.20 × num_products                ← more products = more invested
      + noise
```

This logit score is passed through a sigmoid function to produce a churn probability per customer, then a coin flip per row produces the final 0/1 label. The result is a dataset with meaningful, learnable patterns.

The data is split **80% train / 20% test** with no row overlap — the training set is used to fit the model, the test set is held out and only used for final evaluation.

**Features:**

| Feature | Type | Description |
|---|---|---|
| `tenure` | Numerical | Months as a customer |
| `monthly_charges` | Numerical | Monthly bill amount |
| `total_charges` | Numerical | Cumulative spend |
| `num_products` | Numerical | Number of services subscribed |
| `num_complaints` | Numerical | Complaints filed |
| `days_since_last_login` | Numerical | Engagement indicator |
| `contract` | Categorical | month-to-month / one-year / two-year |
| `payment_method` | Categorical | credit card / bank transfer / etc. |
| `internet_service` | Categorical | fiber / dsl / none |
| `churn` | Target | 0 = stayed, 1 = left |

---

### Model Training (`src/train.py`)

#### The sklearn Pipeline

A `Pipeline` chains preprocessing and model training into a single object. This is important for two reasons:

1. **Convenience** — calling `pipeline.fit(X, y)` runs every step in sequence automatically
2. **Preventing data leakage** — the scaler only learns statistics from training data, never from test data. Without a pipeline it's easy to accidentally fit the scaler on the whole dataset

The preprocessor uses a `ColumnTransformer` to apply different transformations to different column types simultaneously:

- **`StandardScaler`** on numerical features — rescales each column to mean=0, std=1. This prevents the model from misinterpreting scale as importance (e.g. `tenure` ranging 1–72 vs `num_complaints` ranging 0–5)
- **`OneHotEncoder`** on categorical features — converts text categories into binary columns since models can't do arithmetic on strings. `month-to-month` becomes `[1, 0, 0]`, `one-year` becomes `[0, 1, 0]`, and so on

#### XGBoost

XGBoost builds an **ensemble of decision trees** sequentially, where each new tree corrects the mistakes of all the previous ones. Key hyperparameters used:

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 400 | Number of trees — more trees = better fit |
| `max_depth` | 5 | How deep each tree grows — controls complexity |
| `learning_rate` | 0.05 | How aggressively each tree corrects errors — low + many trees generalises better |
| `random_state` | 42 | Ensures reproducible results |

#### Cross-Validation

Rather than evaluating the model on the test set once (which could be lucky or unlucky depending on the split), **5-fold stratified cross-validation** is used on the training data to get a reliable performance estimate.

The training data is divided into 5 equal chunks (folds). In each of 5 rounds, a different fold is held out as a temporary test set while the model trains on the other 4:

```
Round 1: [TEST] [train] [train] [train] [train]  → ROC-AUC score 1
Round 2: [train] [TEST] [train] [train] [train]  → ROC-AUC score 2
Round 3: [train] [train] [TEST] [train] [train]  → ROC-AUC score 3
Round 4: [train] [train] [train] [TEST] [train]  → ROC-AUC score 4
Round 5: [train] [train] [train] [train] [TEST]  → ROC-AUC score 5
                                                    ─────────────
                                          mean ± std  → logged to MLflow
```

**`StratifiedKFold`** is used instead of regular `KFold` because churn is an imbalanced label — most customers don't churn. Stratification ensures each fold preserves the same churn ratio as the full dataset, preventing any fold from having almost no churners.

The mean and standard deviation of the 5 scores are both logged — a high std means the model is unstable across different data slices.

#### Evaluation Metrics

- **ROC-AUC** — measures the model's ability to rank churners above non-churners across all possible thresholds. 0.5 = random, 1.0 = perfect. Better than accuracy for imbalanced data
- **Average Precision** — area under the precision-recall curve. More informative than ROC-AUC when the positive class (churn) is rare

Both are computed using `predict_proba` (probability scores) rather than `predict` (0/1 labels) to capture the full discrimination ability of the model.

#### SHAP Feature Importance

SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for every prediction. Taking the mean absolute SHAP value across all customers gives a global view of which features the model relies on most. The top 5 are printed after training — expected to be `num_complaints`, `contract`, and `tenure` given how the data was generated.

#### MLflow Experiment Tracking

Every training run logs hyperparameters and metrics to MLflow. Run `mlflow ui` to open the dashboard at `http://localhost:5000` and compare runs visually.

---

## Upcoming

### Inference API (`api/main.py`)

A FastAPI server that loads the trained model and serves real-time predictions:

- `POST /predict` — single customer → churn probability, prediction, and risk tier (low / medium / high)
- `POST /predict/batch` — bulk predictions for up to 1000 customers
- `GET /health` — confirms the model is loaded and the server is ready

### Drift Monitoring (`src/monitor.py`)

Uses Evidently to compare the training data distribution against new incoming data and detect when the model's inputs have shifted — a common cause of silent model degradation in production:

- **Data drift report** — flags features whose distribution has changed significantly
- **Classification report** — compares model performance on reference vs current data

Reports are saved as interactive HTML files.

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Generate data
python src/generate_data.py

# Train the model
python src/train.py

# View MLflow runs
mlflow ui
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `scikit-learn` | Pipeline, preprocessing, cross-validation |
| `xgboost` | Gradient boosted tree classifier |
| `mlflow` | Experiment tracking and model registry |
| `shap` | Feature importance explainability |
| `fastapi` + `uvicorn` | Inference API server |
| `evidently` | Data and model drift monitoring |
| `pandas` + `numpy` | Data manipulation |
| `pydantic` | Request validation in the API |
