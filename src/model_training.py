from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os
from config import MODEL_SAVE_PATH, HYPERPARAMS

def train_models(X_train, y_train):
    """Train multiple models with hyperparameter tuning"""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Initialize models with consistent naming
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'xgboost': XGBClassifier(random_state=42),
        'neural_network': MLPClassifier(random_state=42)
    }
    
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=HYPERPARAMS[model_name],  # Use exact key from HYPERPARAMS
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Save best model
        best_model = grid_search.best_estimator_
        trained_models[model_name] = best_model
        joblib.dump(best_model, os.path.join(MODEL_SAVE_PATH, f'{model_name}.joblib'))
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    
    return trained_models