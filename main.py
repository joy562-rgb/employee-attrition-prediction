from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model, save_model
from sklearn.metrics import accuracy_score  # Added this import

DATA_PATH = "data/employee.csv"
MODEL_PATH = "models/attrition_model.pkl"


def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")

    # Save model
    save_model(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    main()