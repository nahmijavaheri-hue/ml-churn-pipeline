# train.py
# Goal: train an XGBoost classifier inside a sklearn Pipeline, tracked with MLflow
#
# Steps to implement:
#   1. Import pandas, sklearn (Pipeline, ColumnTransformer, OneHotEncoder,
#      StandardScaler), xgboost (XGBClassifier), mlflow, shap
#   2. Define your feature lists: NUM_FEATURES, CAT_FEATURES, TARGET
#   3. build_pipeline(): return a Pipeline with two steps:
#        - "pre": ColumnTransformer that scales numerics and one-hot encodes categoricals
#        - "clf": XGBClassifier with whatever hyperparameters you want to try
#   4. main():
#        a. Load data/train.csv and data/test.csv
#        b. Open an mlflow.start_run() context
#        c. Log your hyperparameters with mlflow.log_params()
#        d. Run 5-fold cross-validation and log the mean ROC-AUC
#        e. Fit the pipeline on the full training set
#        f. Evaluate on the test set — log roc_auc and avg_precision
#        g. Use shap.TreeExplainer to print the top feature importances
#        h. Save the model with mlflow.sklearn.save_model() to models/churn_pipeline
