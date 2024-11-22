from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
from utils.preprocessing import preprocess_data
from utils.training import train_model_pipeline
from utils.prediction import make_prediction
import os
import pandas as pd
import json
from flask_session import Session
from config import TARGET_COLUMN, SELECTED_FEATURES

# Initialize Flask application
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Upload and preprocess the file
    """
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded."}), 400
        try:
            file_path = os.path.join("data", file.filename)
            file.save(file_path)
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {e}"}), 500

        # Store file path in session
        session['file_path'] = file_path

        try:
            # Preprocess the data
            cleaned_path, features_path = preprocess_data(file_path, SELECTED_FEATURES, TARGET_COLUMN)
        except Exception as e:
            return jsonify({"error": f"Failed to preprocess file: {e}"}), 500

        # Store cleaned file path in session
        session['cleaned_path'] = cleaned_path
        session['features_path'] = features_path

        return redirect(url_for('upload_success', filename=file.filename))
    return render_template('upload.html')

@app.route('/upload_success/<filename>')
def upload_success(filename):
    """Upload success page"""
    return render_template('upload_success.html', filename=filename)

@app.route('/train', methods=['GET', 'POST'])
def train():
    """
    Train the model
    """
    if request.method == 'POST':
        # Retrieve cleaned file path from session
        cleaned_path = session.get('cleaned_path')
        if not cleaned_path:
            return jsonify({'error': 'No preprocessed file available'}), 400

        try:
            metrics = train_model_pipeline(cleaned_path, TARGET_COLUMN)
        except Exception as e:
            return jsonify({"error": f"Failed to train model: {e}"}), 500

        return redirect(url_for('train_success', metrics=json.dumps(metrics)))
    return render_template('train.html')

@app.route('/train_success')
def train_success():
    """Training success page"""
    try:
        metrics = json.loads(request.args.get('metrics'))
    except Exception as e:
        return jsonify({"error": f"Failed to load training metrics: {e}"}), 500

    features = SELECTED_FEATURES
    target_column = TARGET_COLUMN
    return render_template('train_success.html', metrics=metrics, features=features, target_column=target_column)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Make predictions using the trained model
    """
    if request.method == 'POST':
        # Retrieve cleaned file path from session
        cleaned_path = session.get('cleaned_path')
        if not cleaned_path:
            return jsonify({'error': 'No preprocessed file available'}, 400)

        try:
            # Make prediction
            predictions = make_prediction(cleaned_path)
            predictions_df = pd.DataFrame({
                'ID': range(1, len(predictions) + 1),
                'Prediction': predictions
            })
        except Exception as e:
            return jsonify({"error": f"Failed to make predictions: {e}"}), 500

        session['predictions'] = predictions_df.to_dict(orient='records')

        return redirect(url_for('predict_success'))
    return render_template('predict.html')

@app.route('/predict_success')
def predict_success():
    """Prediction success page"""
    try:
        predictions = session.get('predictions', None)
        if not predictions:
            return "No prediction data available", 400

        df = pd.DataFrame(predictions)
        data = df.to_dict('records')
    except Exception as e:
        return jsonify({"error": f"Failed to load prediction data: {e}"}), 500

    return render_template('predict_success.html', data=data)

@app.route('/get_columns', methods=['POST'])
def get_columns():
    """
    Get columns from the uploaded file
    """
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded."}), 400

        # Save the uploaded file temporarily
        file_path = os.path.join("data/temp", file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

        # Read the dataset and get column names
        data = pd.read_csv(file_path)
        columns = data.columns.tolist()

        # Remove the temporary file after reading
        os.remove(file_path)

        return jsonify({"columns": columns})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_report')
def download_report():
    """Download the report of fraudulent predictions"""
    try:
        data = session.get('predictions', None)
        if data is None:
            return "No prediction data available", 400

        df = pd.DataFrame.from_dict(data)
        fraudulent_df = df[df['Prediction'] == 'Fraudulent']
        report_path = os.path.join('data', 'fraudulent_prediction_report.csv')
        fraudulent_df.to_csv(report_path, index=False)
    except Exception as e:
        return jsonify({"error": f"Failed to generate report: {e}"}), 500

    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

