import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
import pickle
import os
import numpy as np

# Define numerical and categorical features
numerical_features = [
    'vehicle_claim', 'total_claim_amount', 'property_claim', 
    'injury_claim', 'months_as_customer', 'witnesses', 
    'age', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
    'policy_deductable', 'capital-gains', 'capital-loss', 'bodily_injuries'
]

categorical_features = [
    'incident_severity', 'policy_csl', 'collision_type', 
    'incident_state', 'incident_city', 'incident_type', 'insured_occupation',
    'insured_sex', 'property_damage', 'police_report_available',
    'insured_relationship', 'authorities_contacted'
]

# Create a pipeline for numerical features (imputation and scaling)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values by replacing them with the mean
    ('scaler', MinMaxScaler())  # Scale numerical features to a range [0, 1]
])

# Create a pipeline for categorical features (imputation and encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values by replacing them with the most frequent value
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features using one-hot encoding
])

# Combine numerical and categorical transformers into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # Apply numerical transformer to numerical features
        ('cat', categorical_transformer, categorical_features)  # Apply categorical transformer to categorical features
    ]
)

# Create a preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

def preprocess_data(file_path, selected_features, target_column):
    """
    Preprocess the data and return the path to the cleaned file
    
    Parameters:
    - file_path: str, path to the raw data file
    - selected_features: list, features to be preprocessed
    - target_column: str, target column name
    
    Returns:
    - cleaned_path: str, path to the cleaned data file
    - features_path: str, path to the pickle file containing feature names
    """
    try:
        # Load the data from the file
        data = pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error reading the file: {file_path}. Exception: {e}")
    
    data.replace('?', np.nan, inplace=True)  # Replace '?' with NaN for proper handling of missing values

    try:
        # Select the specified features for preprocessing
        data_selected = data[selected_features]
    except KeyError as e:
        raise KeyError(f"Error selecting features: {e}")

    try:
        # Encode the target column if it is categorical
        label_encoder = LabelEncoder()
        data[target_column] = label_encoder.fit_transform(data[target_column])
    except KeyError as e:
        raise KeyError(f"Error encoding target column: {e}")

    try:
        # Apply the preprocessing pipeline to the selected features
        data_processed = preprocessing_pipeline.fit_transform(data_selected)
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

    # Capture the new column names after transformation
    preprocessed_columns = numerical_features + list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features))

    # Ensure the processed data is a numpy array
    if not isinstance(data_processed, np.ndarray):
        raise ValueError("Processed data is not a numpy array")

    try:
        # Convert the processed data to a DataFrame with new column names
        data_processed_df = pd.DataFrame(data_processed, columns=preprocessed_columns)
    except ValueError as e:
        raise ValueError(f"Error creating DataFrame: {e}, Data shape: {data_processed.shape}, Columns: {len(preprocessed_columns)}")

    # Add the encoded target column back to the DataFrame
    data_processed_df[target_column] = data[target_column].values

    # Save the cleaned data to a new file
    try:
        cleaned_path = os.path.join('data', 'insuranceFraud_cleaned.csv')
        data_processed_df.to_csv(cleaned_path, index=False)
    except Exception as e:
        raise IOError(f"Error saving cleaned data to file: {cleaned_path}. Exception: {e}")

    # Save the selected feature names to a pickle file
    try:
        features_path = os.path.join('models', 'feature_names.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(preprocessed_columns, f)
    except Exception as e:
        raise IOError(f"Error saving feature names to pickle file: {features_path}. Exception: {e}")

    return cleaned_path, features_path

