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
import shutil

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

def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    X_train = train[NUM_FEATURES + CAT_FEATURES]
    y_train = train[TARGET]

    X_test = test[NUM_FEATURES + CAT_FEATURES]
    y_test = test[TARGET]

    mlflow.set_experiment("churn-prediction")
    with mlflow.start_run():
        mlflow.log_params({"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05})
        pipe = build_pipeline()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")

        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:,1]
        roc_auc = roc_auc_score(y_test, y_prob)
        average_precision = average_precision_score(y_test, y_prob)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("average_precision", average_precision)

        explainer = shap.TreeExplainer(pipe.named_steps["clf"])
        X_transformed = pipe.named_steps["pre"].transform(X_test)
        shap_values = explainer.shap_values(X_transformed)

        feature_names = (
            NUM_FEATURES
            + pipe.named_steps["pre"].named_transformers_["cat"].
            get_feature_names_out(CAT_FEATURES).tolist()
        )

        mean_shap_values = np.abs(shap_values).mean(axis=0)
        top_five = sorted(zip(feature_names, mean_shap_values), key=lambda x: -x[1])[:5]
        for feature, shap_value in top_five:
            print(f"{feature}: {shap_value:.4f}")

        MODELS.mkdir(exist_ok=True)
        shutil.rmtree(MODELS / "churn_pipeline", ignore_errors=True)
        mlflow.sklearn.save_model(pipe, MODELS / "churn_pipeline")
if __name__ == "__main__":
    main()

