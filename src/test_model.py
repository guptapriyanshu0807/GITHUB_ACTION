import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import joblib
import json
import pytest

def test_model_prediction():
    # Load the iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Load the trained model
    model = joblib.load('model.joblib')

    # Make predictions on the dataset
    predictions = model.predict(X)

    # Basic va`lidation checks
    assert len(predictions) == len(y), "Predictions length does not match target length"
    
def test_model_accuracy():
    # Load the iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Load the trained model
    model = joblib.load('model.joblib')

    # Calculate accuracy
    accuracy = model.score(X, y)
    
    # Check if accuracy is above a threshold
    assert accuracy > 0.8  , f"Model accuracy {accuracy} is below the expected threshold of 0.8"

