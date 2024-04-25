import subprocess

# List of packages to install
packages = ['scipy', 'scikit-learn', 'seaborn']

# Function to install packages
def install_packages(packages):
    for package in packages:
        subprocess.call(['pip', 'install', package])

# Install packages
install_packages(packages)

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
import streamlit as st

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
 
# Load and preprocess data
x, y = load_data_and_preprocess(data)

# Split data into train and test sets
x_train, x_test, y_train, y_test = split_data(x, y)

# Scale data using RobustScaler
x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

# Train Logistic Regression model
logreg = train_logistic_regression(x_train_scaled, y_train)

# Evaluate Logistic Regression model
#evaluate_logistic_regression(logreg, x_test_scaled, y_test)

# ------------------------------------------------------

# Streamlit App
st.title('Expresso Churn Prediction Challenge')
st.subheader('Predicting Customer Churn Probability')
st.write("Expresso is an African telecommunications services company operating in Mauritania and Senegal.")

status = st.radio('Start prediction:', ('No', 'Yes'))
if status == "Yes":

    # Map original region names to coded values
    region_mapping = {'DAKAR': 0, 'THIES': 12, 'SAINT-LOUIS': 9, 'LOUGA': 7, 'KAOLACK': 4, 
                      'DIOURBEL': 1, 'TAMBACOUNDA': 11, 'KAFFRINE': 3, 'KOLDA': 6, 
                      'FATICK': 2, 'MATAM': 8, 'ZIGUINCHOR': 13, 'SEDHIOU': 10, 'KEDOUGOU': 5}

    # User selects a region
    selected_region_original = st.selectbox("Select your region:", list(region_mapping.keys()))

    # Map the selected original region to its coded value
    selected_region_coded = region_mapping[selected_region_original]
    
    # Map original tenure names to coded values
    tenure_mapping = {'K > 24 month': 7, 'I 18-21 month': 5, 'H 15-18 month': 4, 
                      'G 12-15 month': 3, 'J 21-24 month': 6, 'F 9-12 month': 2, 
                      'E 6-9 month': 1, 'D 3-6 month': 0}

    # User selects a tenure
    selected_tenure_original = st.selectbox("Select your subscription tenure:", list(tenure_mapping.keys()))

    # Map the selected original tenure to its coded value
    selected_tenure_coded = tenure_mapping[selected_tenure_original]

    # Create DataFrame for region and tenure
    region_tenure_data = pd.DataFrame({'REGION': [selected_region_coded], 'TENURE': [selected_tenure_coded]})
    
    # Input fields for other features
    st.subheader("Enter other details:")
    others_features = {'MONTANT': 0, 'FREQUENCE_RECH': 0, 'REVENUE': 0,
                      'ARPU_SEGMENT': 0, 'FREQUENCE': 0, 'DATA_VOLUME': 0, 'ON_NET': 0, 'ORANGE': 0,
                      'TIGO': 0, 'ZONE1': 0, 'ZONE2': 0, 'REGULARITY': 0, 'TOP_PACK': 0,
                      'FREQ_TOP_PACK': 0}
    data = {}
    for key in others_features:
        value = st.number_input(f"Enter value for {key}: ")
        data[key] = value

    # Create DataFrame for other features
    input_data = pd.DataFrame(data, index=[0])

    # Concatenate DataFrames
    test_df = pd.concat([region_tenure_data, input_data], axis=1)
    test_df['MRG'] = 0

    st.dataframe(test_df)

    if st.button('Predict'):
        # Predict using the model
        prediction = logreg.predict(test_df.values)
        if prediction == 1:
            st.warning('Prediction: The model predicts the customer is likely to churn.')
        else:
            st.success('Prediction: The customer is unlikely to churn.')

