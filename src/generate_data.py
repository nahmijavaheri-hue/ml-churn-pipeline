# generate_data.py
# Goal: create a synthetic customer dataset and save train/test CSVs to data/
#
# Steps to implement:
#   1. Import numpy, pandas, and pathlib.Path
#   2. Define the output folder (data/) relative to this file
#   3. Write a generate(n) function that builds a DataFrame with columns like:
#        - tenure (int, months as a customer)
#        - monthly_charges (float)
#        - total_charges (float)
#        - num_products (int)
#        - num_complaints (int)
#        - days_since_last_login (int)
#        - contract (categorical: month-to-month / one-year / two-year)
#        - payment_method (categorical)
#        - internet_service (categorical)
#        - churn (0 or 1) — this is your TARGET column
#      Tip: churn should not be random — make it depend on the other columns
#      using a logistic formula so the model has something real to learn
#   4. In __main__: call generate(), split 80/20, save train.csv and test.csv

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent.parent / "data"

def generate(n):
    df = pd.DataFrame({})