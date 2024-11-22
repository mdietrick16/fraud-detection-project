import pandas as pd
import pickle

def load_model(model_path="models/trained_model.pkl"):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise ValueError("Model file not found. Please train the model first.")

def load_feature_names(feature_path = 'models/feature_names.pkl'):
    try:
        with open(feature_path, "rb") as f:
            feature_names = pickle.load(f)
        return feature_names
    except FileNotFoundError:
        raise ValueError("Feature names file not found. Please train the model first.")

def make_prediction(file_path):
    try:
        # Load the trained model and feature names
        model = load_model()
        feature_names = load_feature_names()

        # Load and preprocess the input data
        data = pd.read_csv(file_path)

        # Ensure the input data has the required features
        data = data[feature_names]

        # Make predictions
        predictions = model.predict(data)
        prediction_labels = ["Legitimate" if pred == 0 else "Fraudulent" for pred in predictions]

        return prediction_labels
    except Exception as e:
        raise ValueError(f"Error making predictions: {e}")
