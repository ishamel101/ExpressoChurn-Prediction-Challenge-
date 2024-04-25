import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
import seaborn as sns


# Load the data
data = pd.read_csv('Expresso_churn_dataset.csv', sep=",")

# Handle missing values
def handle_missing_values(df):
    # Fill missing numerical values with mean
    numerical_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 
                      'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
                      'ZONE1', 'ZONE2', 'FREQ_TOP_PACK']
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = ['TOP_PACK', 'REGION']
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

# Remove outliers using z-score
def remove_outliers_zscore(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(exclude="object")))
    df_no_outliers = df[(z_scores < 3).all(axis=1)]
    return df_no_outliers

# Encode categorical variables
def encode_categorical_variables(df):
    object_coding = ["REGION", "TENURE", "MRG", "TOP_PACK"]
    encoder = LabelEncoder()
    for col in object_coding:
        df[col] = encoder.fit_transform(df[col])
    return df

# Apply data preprocessing steps
data = handle_missing_values(data)
data = remove_outliers_zscore(data)
data = encode_categorical_variables(data)

# Machine Learning Functions

# Load data and preprocess
def load_data_and_preprocess(data):
    # Extract features and target variable
    x = data[['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
                'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
                'ZONE1', 'ZONE2', 'MRG', 'REGULARITY', 'TOP_PACK', 'FREQ_TOP_PACK']]
    y = data['CHURN'].values
    
    return x, y

# Split data
def split_data(x, y, test_size=0.35, random_state=100):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    return x_train, x_test, y_train, y_test

# Scale data
def scale_data(x_train, x_test, scaler_type='Robust'):
    if scaler_type == 'Robust':
        scaler = RobustScaler()
    elif scaler_type == 'Standard':
        scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    return x_train_scaled, x_test_scaled

# Train Logistic Regression Model
def train_logistic_regression(x_train_scaled, y_train):
    # Logistic Regression Model
    logreg = LogisticRegression()
    logreg.fit(x_train_scaled, y_train)
    
    return logreg

# Evaluate Logistic Regression model
def evaluate_logistic_regression(logreg, x_test_scaled, y_test):
    # Model Evaluation - Logistic Regression
    y_pred_logreg = logreg.predict(x_test_scaled)
    logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
    print("Logistic Regression Accuracy:", logreg_accuracy)
    
    # Confusion Matrix - Logistic Regression
    confusion_matrix_logreg = pd.crosstab(y_test, y_pred_logreg, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix_logreg, annot=True)

# Train Random Forest Classifier
def train_random_forest_classifier(x_train, y_train, n_estimators=45, random_state=42):
    # Random Forest Classifier
    rnf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rnf_classifier.fit(x_train, y_train)
    
    return rnf_classifier

# Evaluate Random Forest Classifier
def evaluate_random_forest_classifier(rnf_classifier, x_test, y_test):
    # Model Evaluation - Random Forest Classifier
    y_pred_rnf = rnf_classifier.predict(x_test)
    rnf_accuracy = accuracy_score(y_test, y_pred_rnf)
    print("Random Forest Classifier Accuracy:", rnf_accuracy)

# Load and preprocess data
x, y = load_data_and_preprocess(data)

# Split data into train and test sets
x_train, x_test, y_train, y_test = split_data(x, y)

# Scale data using RobustScaler
x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

# Train Logistic Regression model
logreg = train_logistic_regression(x_train_scaled, y_train)

# Evaluate Logistic Regression model
evaluate_logistic_regression(logreg, x_test_scaled, y_test)

# Scale data using StandardScaler for Random Forest Classifier
X_train, X_test = scale_data(x_train, x_test, scaler_type='Standard')

# Train Random Forest Classifier model
rnf_classifier = train_random_forest_classifier(X_train, y_train)

# Evaluate Random Forest Classifier model
evaluate_random_forest_classifier(rnf_classifier, X_test, y_test)


# # Save the LOGISCTIC Trained Model
# import joblib
# joblib.dump(logreg, 'Logistic_model_predection.joblib')

# # Save the RANDOM FOREST Trained Model
# import joblib
# joblib.dump(rnf_classifier, 'RandomForest_model_predection.joblib')

