import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from scipy.sparse import issparse
from config import DATA_PATH, PROCESSED_DATA_PATH, TEST_SIZE, RANDOM_STATE

def parse_response_time(x):
    """Parse Response_Time field handling different formats"""
    try:
        if '-' in x:  # Format like "5-15 mins"
            parts = x.split('-')
            lower = int(parts[0])
            upper = int(parts[1].split()[0])
            return (lower + upper) / 2
        elif x.startswith('<'):  # Format like "<5 mins"
            return int(x[1:].split()[0])
        elif x.startswith('>'):  # Format like ">15 mins"
            return int(x[1:].split()[0])
        else:  # Single number format
            return int(x.split()[0])
    except (ValueError, IndexError):
        return 0  # Default value if parsing fails

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    # Convert Response_Time to numerical using the parsing function
    df['Response_Time'] = df['Response_Time'].apply(parse_response_time)
    
    # Separate features and target
    X = df.drop('Alert_Level', axis=1)
    y = df['Alert_Level']
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Define categorical and numerical features
    categorical_features = ['Sensor_Type', 'Incident_Type', 'Weather_Condition', 
                          'Time_of_Day', 'Traffic_Density', 'Detection_Method', 
                          'Location_Type', 'Data_Source']
    numerical_features = ['Response_Time']
    
    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    # Convert to dense array if sparse
    if issparse(X_processed):
        X_processed = X_processed.toarray()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Save processed data and encoders
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(PROCESSED_DATA_PATH, 'train_features.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(PROCESSED_DATA_PATH, 'test_features.csv'), index=False)
    pd.Series(y_train).to_csv(os.path.join(PROCESSED_DATA_PATH, 'train_labels.csv'), index=False)
    pd.Series(y_test).to_csv(os.path.join(PROCESSED_DATA_PATH, 'test_labels.csv'), index=False)
    
    # Save preprocessor and label encoder
    joblib.dump(preprocessor, os.path.join(PROCESSED_DATA_PATH, 'preprocessor.joblib'))
    joblib.dump(label_encoder, os.path.join(PROCESSED_DATA_PATH, 'label_encoder.joblib'))
    
    return X_train, X_test, y_train, y_test, preprocessor, label_encoder