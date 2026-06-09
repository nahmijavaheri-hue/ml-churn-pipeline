# train.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import shap
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
MODELS = Path(__file__).parent.parent / "models"

NUM_FEATURES = ["tenure", "monthly_charges", "total_charges",
                "num_products", "num_complaints", "days_since_last_login"]
CAT_FEATURES = ["contract", "payment_method", "internet_service"]
TARGET = "churn"

def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])
    clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])



