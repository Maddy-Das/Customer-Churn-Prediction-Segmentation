import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing import preprocess_data
from src.models import train_classification_models
from src.utils import plot_confusion  # Optional, if you have this function

def main():
    # Step 1: Load data
    try:
        df = pd.read_csv("../../data/customer_data.csv")
        print(f"✅ Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: '../../data/customer_data.csv' not found.")
        return

    # Step 2: Preprocess data
    try:
        X, y = preprocess_data(df)
        print(f"✅ Preprocessed data. Features shape: {X.shape}, Target shape: {y.shape}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Split data into train and test sets.")

    # Step 4: Train models
    models = train_classification_models(X_train, y_train)
    print("✅ Trained classification models.")

    # Step 5: Evaluate models
    print("\n📊 Model Evaluation:")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n🔍 {name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # Optional: Confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion(cm, title=f"{name} Confusion Matrix")
        except Exception as e:
            print(f"(i) Skipped confusion matrix for {name}: {e}")

if __name__ == "__main__":
    main()
