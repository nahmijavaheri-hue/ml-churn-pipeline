# generate_data.py

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent.parent / "data"

RNG = np.random.default_rng(42)


def generate(n):
    customer_id = [f"C{i:06d}" for i in range(n)]
    tenure = RNG.integers(1, 73, n)
    monthly_charges = RNG.uniform(20, 120, n).round(2)
    total_charges = (monthly_charges * tenure + RNG.normal(0, 50, n)).clip(0).round(2)
    num_products = RNG.integers(1, 6, n)
    num_complaints = RNG.poisson(0.3, n)
    days_since_last_login = RNG.integers(0, 90, n)
    contract = RNG.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    payment_method = RNG.choice(["credit_card", "bank_transfer", "electronic_check", "mailed_check"], n)
    internet_service = RNG.choice(["fiber", "dsl", "none"], n, p=[0.44, 0.34, 0.22])
    logit = (
        -3.5
        + 0.04 * (monthly_charges - 65)       # high charges → more churn
        - 0.03 * tenure                        # long tenure → less churn
        + 0.5  * num_complaints                # complaints → strong churn signal
        + 0.02 * days_since_last_login         # disengaged → more churn
        + 0.8  * (contract == "month-to-month")# easy to leave → more churn
        - 0.5  * (contract == "two-year")      # locked in → less churn
        + 0.3  * (internet_service == "fiber") # fiber customers churn more (expensive)
        - 0.2  * num_products                  # more products → more invested → less churn
        + RNG.normal(0, 0.3, n)                # small random noise
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (RNG.uniform(0, 1, n) < churn_prob).astype(int)


    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_products": num_products,
        "num_complaints": num_complaints,
        "days_since_last_login": days_since_last_login,
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "churn": churn,
    })
    return df

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df1 = generate(10000)
    train = df1.sample(frac=0.8, random_state=42)
    test = df1.drop(train.index)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
