from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from config import MODEL_SAVE_PATH

def evaluate_models(models, X_test, y_test, label_encoder):
    """Evaluate trained models on test set"""
    evaluation_results = {}
    
    # Convert encoded labels back to original for reporting
    y_test_original = label_encoder.inverse_transform(y_test)
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_original = label_encoder.inverse_transform(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test_original, y_pred_original)
        
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'classification_report': report
        }
        
        # Save evaluation results
        with open(os.path.join(MODEL_SAVE_PATH, f'{model_name}_report.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("Classification Report:\n")
            f.write(report)  # Now writing string instead of dict
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test_original, y_pred_original, 
                             labels=label_encoder.classes_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(MODEL_SAVE_PATH, f'{model_name}_confusion_matrix.png'))
        plt.close()
    
    return evaluation_results