from src.data_processing import load_and_preprocess_data
from src.model_training import train_models
from src.evaluation import evaluate_models

def main():
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = load_and_preprocess_data()
    
    # Step 2: Train models
    print("Training models...")
    trained_models = train_models(X_train, y_train)
    
    # Step 3: Evaluate models
    print("Evaluating models...")
    evaluation_results = evaluate_models(trained_models, X_test, y_test, label_encoder)
    
    # Print results
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Classification Report:")
        print(results['classification_report'])
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()