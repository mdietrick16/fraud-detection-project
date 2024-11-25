import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # done towards the end to show only two features in graphs

def load_data(file_path):
    """
    Load the dataset from file path
    
    Parameters:
    - file_path: str, path to the data file
    
    Returns:
    - data: DataFrame, loaded dataset
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error reading the file: {file_path}. Exception: {e}")
    
    return data

def preprocess_data(data, target_column):
    """
    Separate features and target from the dataset
    
    Parameters:
    - data: DataFrame, the dataset
    - target_column: str, name of the target column
    
    Returns:
    - X: DataFrame, features
    - y: Series, target
    """
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    except KeyError as e:
        raise KeyError(f"Error separating features and target column '{target_column}': {e}")
    
    return X, y

def handle_imbalance(X, y):
    """
    Handle class imbalance using SMOTE
    
    Parameters:
    - X: DataFrame, features
    - y: Series, target
    
    Returns:
    - X_resampled: DataFrame, resampled features
    - y_resampled: Series, resampled target
    """
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except Exception as e:
        raise ValueError(f"Error handling class imbalance with SMOTE: {e}")
    
    return X_resampled, y_resampled

def split_data(X, y):
    """
    Split dataset into training and testing sets
    
    Parameters:
    - X: DataFrame, features
    - y: Series, target
    
    Returns:
    - X_train: DataFrame, training features
    - X_test: DataFrame, testing features
    - y_train: Series, training target
    - y_test: Series, testing target
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    except Exception as e:
        raise ValueError(f"Error splitting data into training and testing sets: {e}")
    
    return X_train, X_test, y_train, y_test

def define_models():
    """
    Define base models and meta-learners for stacking
    
    Returns:
    - base_models: list of tuples, base models for stacking
    - meta_learner: classifier, meta-learner for stacking
    """
    try:
        base_models = [
            ('gb', GradientBoostingClassifier(
                n_estimators=150, 
                learning_rate=0.1, 
                max_depth=3, 
                random_state=42)),
            ('knn', KNeighborsClassifier(
                n_neighbors=1))
        ]
        meta_learner = LogisticRegression(
            max_iter=1000)
    except Exception as e:
        raise ValueError(f"Error defining models: {e}")
    
    return base_models, meta_learner

def train_model(X_train, y_train, base_models, meta_learner):
    """
    Train a stacking ensemble model
    
    Parameters:
    - X_train: DataFrame, training features
    - y_train: Series, training target
    - base_models: list of tuples, base models for stacking
    - meta_learner: classifier, meta-learner for stacking
    
    Returns:
    - model: StackingClassifier, trained stacking model
    """
    try:
        model = StackingClassifier(estimators=base_models, final_estimator=meta_learner)
        model.fit(X_train, y_train)
    except Exception as e:
        raise ValueError(f"Error training stacking ensemble model: {e}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    
    Parameters:
    - model: StackingClassifier, trained stacking model
    - X_test: DataFrame, testing features
    - y_test: Series, testing target
    
    Returns:
    - metrics: dict, classification metrics
    """
    try:
        y_pred = model.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True)
    except Exception as e:
        raise ValueError(f"Error evaluating model: {e}")
    
    return metrics

def save_model(model, model_path):
    """
    Save the trained model to a file
    
    Parameters:
    - model: StackingClassifier, trained stacking model
    - model_path: str, path to save the model file
    """
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        raise IOError(f"Error saving model to file: {model_path}. Exception: {e}")
    
def present_models(X_train, y_train, X):
    '''
    To better present, add plots to show KNN and GradientBoosting and compare with them stacked.
    Parameters:
         X_train, y_train that have been split
         X from dataset
    Returns:
         No returns but png files are created for graphs.
    '''
    #define and train the classifiers
    gb_classifier = GradientBoostingClassifier()
    knn_classifier = KNeighborsClassifier()
    stacking_classifier = StackingClassifier(estimators=[
        ('gb', gb_classifier),
        ('knn', knn_classifier)],
        final_estimator=LogisticRegression()
    )
    # We used 15 features and should reduce it down to two, so for the graph PCA to reduce it to 2
    pca = PCA(n_components=2) #Principal Component Analysis retaining variance while reducing dimensionality
                               # This will allow us to visualize the first two principal components.
    X_train_pca = pca.fit_transform(X_train)

    gb_classifier.fit(X_train_pca, y_train)
    knn_classifier.fit(X_train_pca, y_train)
    stacking_classifier.fit(X_train_pca, y_train)

    #create a mesh grid for plotting decision boundaries
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  
    #get predictions for the mesh grid
    gb_predictions = gb_classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    knn_predictions = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    stacking_predictions = stacking_classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plot_classifier("Gradient Boosting Classifier", gb_predictions, X_train_pca, y_train,xx,yy)
    plot_classifier("K-Nearest Neighbors Classifier", knn_predictions, X_train_pca, y_train,xx,yy)
    plot_classifier("Stacking Classifier", stacking_predictions, X_train_pca, y_train,xx,yy)

def plot_classifier(classifier_name, predictions, X_train, y_train,xx,yy):
    plt.figure(figsize=(8,6))
    plt.contour(xx, yy, predictions, alpha=0.5, cmap='coolwarm')
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, edgecolors='k', marker='o')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(classifier_name)
    plt.savefig(f'{classifier_name.replace(" ", "_").lower()}_decision_boundary.png', dpi=300)
    plt.close()

def train_model_pipeline(cleaned_path, target_column):
    """
    Full training pipeline
    
    Parameters:
    - cleaned_path: str, path to the cleaned data file
    - target_column: str, name of the target column
    
    Returns:
    - metrics: dict, classification metrics of the trained model
    """
    try:
        data = load_data(cleaned_path)
        X, y = preprocess_data(data, target_column)
        X_resampled, y_resampled = handle_imbalance(X, y)
        X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
        base_models, meta_learner = define_models()
        model = train_model(X_train, y_train, base_models, meta_learner)
        metrics = evaluate_model(model, X_test, y_test)
        model_path = os.path.join('models', 'trained_model.pkl')
        #added to show graphs for each classifier
        matplotlib.use("Agg")  # Use the Agg backend for saving plots
        present_models(X_train, y_train, X)
        save_model(model, model_path)
    except Exception as e:
        raise RuntimeError(f"Error in training pipeline: {e}")
    
    return metrics

