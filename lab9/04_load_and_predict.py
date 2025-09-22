import mlflow
from sklearn.datasets import load_wine
import numpy as np

def load_and_predict():
    MODEL_NAME = "wine-classifier-prod"
    stage = "staging"

    print(f"Loading model '{MODEL_NAME}' with stage '{stage}'...")

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{stage}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    wine = load_wine()
    X = wine.data
    y = wine.target

    sample_data = X[0:1]
    actual_label = y[0]

    prediction = model.predict(sample_data)

    print("-" * 30)
    print(f"Sample Data Features:\n{sample_data[0]}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {prediction[0]}")
    print("-" * 30)

if __name__ == "__main__":
    load_and_predict()
