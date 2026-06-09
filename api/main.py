# main.py  (FastAPI inference server)
# Goal: load the trained model and serve predictions over HTTP
#
# Steps to implement:
#   1. Import FastAPI, pydantic BaseModel, mlflow.sklearn, pandas
#   2. Define a Customer input schema (Pydantic model) matching your feature columns
#      Add validation (e.g. tenure >= 0, contract must be one of three values)
#   3. Load the model at startup using FastAPI's lifespan context manager
#      (avoids reloading on every request)
#   4. POST /predict — accepts a single Customer, returns:
#        - churn_probability (float)
#        - churn_prediction (bool, threshold 0.5)
#        - risk_tier ("low" / "medium" / "high")
#   5. POST /predict/batch — same but accepts a list of Customers
#   6. GET /health — returns whether the model is loaded
