import joblib
import pandas as pd
from config import PROCESSED_DATA_PATH, MODEL_SAVE_PATH

def load_data():
    """Load processed data"""
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'train_features.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'test_features.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'train_labels.csv'), squeeze=True)
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'test_labels.csv'), squeeze=True)
    
    return X_train, X_test, y_train, y_test

def load_models():
    """Load trained models"""
    models = {}
    for model_file in os.listdir(MODEL_SAVE_PATH):
        if model_file.endswith('.joblib'):
            model_name = model_file.split('.')[0]
            models[model_name] = joblib.load(os.path.join(MODEL_SAVE_PATH, model_file))
    
    return models