# Configuration file for the project

DATA_PATH = "data/raw/traffic_incidents.csv"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_SAVE_PATH = "models/trained_models/"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hyperparameters for models
HYPERPARAMS = {
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'xgboost': {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6],
        'n_estimators': [100, 200]
    },
    'neural_network': {
        'hidden_layer_sizes': [(64,), (128, 64)],
        'activation': ['relu'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500],  # Increased from default 200
        'early_stopping': [True]  # Add early stopping
    }
}