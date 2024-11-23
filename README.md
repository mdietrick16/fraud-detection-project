
```markdown
# Fraud Detection Project

Welcome to the **Fraud Detection Project**! This is a Flask-based web application designed to help you upload data, preprocess it, train a model, and make predictions for fraud detection specifically for auto insurance fraud detection using insuranceFraud.csv from kaggle.  It's a beginner-friendly tool for exploring machine learning workflows.

---

## Features

1. **Upload Raw Data**: 
   - Upload your dataset, and the application will preprocess it for you (insurance.csv).
   - Preprocessing includes handling missing values and encoding categorical data.

2. **Train a Model**:
   - Use the cleaned dataset to train a machine learning model.
   - The target column is specifically fraud_reported as listed in config.py
   - The selected features are also in config.py and those can be changed accordingly.

3. **Make Predictions**:
   - Use the cleaned dataset to make predictions (e.g., fraudulent or legitimate cases).
   - Predictions are displayed in JSON format for easy interpretation.

4. **Simple Navigation**:
   - Initially the home page leads to the upload page.
   - Second page allows the user to choose the file and upload.
   - Once successful, the metrics and values results are shown.  Features and target column are also displayed.
   - The user can continue to predict in which the ID and the prediction for Fraud are displayed.
   - Lastly, the user is able to download the report. If chosen, fradulent_prediction_report.csv is downloaded to the user's download folder.

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
- Go to the **Upload Data** page by clicking Upload Data
- Upload your raw dataset (e.g., `insurance_fraud.csv`) and click Upload.
- The app will clean the data and save it in the `data` subfolder.
- Cleaned data is called insuranceFraud_cleaned.csv

### 2. **Train Model**
- Go to the **Train Model** page by clicking the Train the Model button.
- Click Start Training to train the machine learning model.
- The training process outputs metrics like accuracy, precision, recall, and F1-score as well as the value.  Also in the results are macro avg as well as weighted avg of said metrics.
- Features Used for Training are also listed on the page.

### 3. **Make Predictions**
- Go to the **Predict** page by clicking the Predict button from the Training Page.
- Click on Make Prediction in the subsequent page.
- The app will output predictions, showing whether each row is "Fraudulent" or "Legitimate".

---

## Notes

1. **Preprocessing**:
   - Always upload the raw dataset to clean it before training or prediction.
   - The raw data is cleaned and modified to include _clean to be updated as the trained model 

2. **Consistent Data**:
   - The cleaned dataset is used for both training and prediction to avoid errors.

3. **Error Handling**:
   - If you encounter any issues (e.g., file not uploading or prediction errors), check the Flask terminal output for error messages.

4. **Flexibility**:
   - This project uses both the GradiantBoostingClassifier in combination of K nearest neighbor, but you can modify the `training.py` file to experiment with other models.

---

## Future Improvements

- Add data visualization features.
- Provide downloadable results for predictions.
- Include model comparison for different algorithms.

---

Enjoy exploring machine learning workflows with this project! 😊
```
