# monitor.py
# Goal: use Evidently to detect data drift between training data and new data
#
# Steps to implement:
#   1. Import evidently — Report, DataDriftPreset, ClassificationPreset, ColumnMapping
#   2. Load data/train.csv (reference) and data/test.csv (current)
#   3. Define a ColumnMapping telling Evidently which columns are:
#        - the target, numerical features, categorical features
#   4. Build a drift Report with DataDriftPreset and run it
#      Save the HTML report to reports/data_drift.html
#   5. Load the trained model, generate predictions on both datasets,
#      then run a ClassificationPreset report comparing them
#      Save to reports/classification.html
