
```markdown
# Fraud Detection Project

Welcome to the **Fraud Detection Project**! This is a Flask-based web application designed to help you upload data, preprocess it, train a model, and make predictions for fraud detection. It's a beginner-friendly tool for exploring machine learning workflows.

---

## Features

1. **Upload Raw Data**: 
   - Upload your dataset, and the application will preprocess it for you.
   - Preprocessing includes handling missing values and encoding categorical data.

2. **Train a Model**:
   - Use the cleaned dataset to train a machine learning model.
   - Select the target column and train a model with metrics displayed after training.

3. **Make Predictions**:
   - Use the cleaned dataset to make predictions (e.g., fraudulent or legitimate cases).
   - Predictions are displayed in JSON format for easy interpretation.

4. **Simple Navigation**:
   - A home menu links to all functionalities: Upload Data, Train Model, and Predict.

---

## Requirements

To run this project, you need the following dependencies installed. Use the `requirements.txt` file to set up your environment:

```bash
pip install -r requirements.txt
```

### Dependencies
- Flask
- pandas
- scikit-learn
- pickle
- gunicorn (optional for production)

---

## How to Use

### 1. Clone or Download the Repository
Clone the repository or download the files to your local machine.

```bash
git clone git@github.com:Elmessaoudih/Fraud-Detection-Project.git
cd fraud_detection_project
```

### 2. Set Up the Environment
Make sure Python 3.x is installed. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask server:

```bash
python app.py
```

The application will run on `http://127.0.0.1:5000`.

---

## Workflow

### 1. **Upload Data**
- Go to the **Upload Data** page.
- Upload your raw dataset (e.g., `insurance_fraud.csv`).
- The app will preprocess the data and save it in the `data/cleaned_data` folder.
- You’ll receive the path to the cleaned dataset.

### 2. **Train Model**
- Go to the **Train Model** page.
- Upload the cleaned dataset and select the target column (e.g., `fraud_reported`).
- Click "Train Model" to train the machine learning model.
- The training process outputs metrics like accuracy, precision, recall, and F1-score.

### 3. **Make Predictions**
- Go to the **Predict** page.
- Upload the cleaned dataset (with the same format as the training data).
- The app will output predictions, showing whether each row is "Fraudulent" or "Legitimate".

---

## Notes

1. **Preprocessing**:
   - Always upload the raw dataset to preprocess it before training or prediction.

2. **Consistent Data**:
   - The cleaned dataset must be used for both training and prediction to avoid errors.

3. **Error Handling**:
   - If you encounter any issues (e.g., file not uploading or prediction errors), check the Flask terminal output for error messages.

4. **Flexibility**:
   - This project uses a Random Forest model, but you can modify the `training.py` file to experiment with other models.

---

## Future Improvements

- Add data visualization features.
- Provide downloadable results for predictions.
- Include model comparison for different algorithms.

---

Enjoy exploring machine learning workflows with this project! 😊
```
