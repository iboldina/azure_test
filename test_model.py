import joblib
import pytest
import pandas as pd
def test_output_format():
    # Load the model from the joblib file
    model = joblib.load('model.joblib')

    # Load the test data from the CSV file
    X_test = pd.read_csv('test_production.csv')

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Check if y_pred is not empty
    assert len(y_pred) > 0, "The output is empty"

    # Check if y_pred only contains expected class labels (e.g., 0 and 1)
    assert set(y_pred) <= {0, 1}, "The output contains unexpected class labels"
