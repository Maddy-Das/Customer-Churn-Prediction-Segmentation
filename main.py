import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing import preprocess_data
from src.models import train_classification_models
from src.utils import plot_confusion

# 🔧 Helper function to plot metric comparisons
def plot_comparison(metrics_dict, title, ylabel):
    plt.figure(figsize=(8, 5))
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    plt.bar(names, values, color='skyblue')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

def main():
    # 📥 Load data
    try:
        df = pd.read_csv("data/customer_data.csv")
        print(f"✅ Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: 'data/customer_data.csv' not found.")
        return

    # ⚙️ Preprocess
    try:
        X, y, X_scaled = preprocess_data(df)
        print(f"✅ Preprocessed data. Features shape: {X_scaled.shape}, Target shape: {y.shape}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return

    # ✂️ Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("✅ Split data into train and test sets.")

    # 🧠 Train models
    models, metrics = train_classification_models(X_train, y_train)
    print("✅ Trained classification models.")

    # 📊 Evaluate
    print("\n📊 Model Evaluation:")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metrics["Accuracy"][name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # Optional: Confusion Matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion(cm, model_name=name)
        except Exception as e:
            print(f"(i) Skipped confusion matrix for {name}: {e}")

    # 📈 Plot comparisons
    try:
        plot_comparison(metrics["Accuracy"], "Model Accuracy Comparison", "Accuracy")
        plot_comparison(metrics["Memory"], "Model Memory Usage Comparison (MB)", "Memory (MB)")
        plot_comparison(metrics["Time"], "Model Training Time Comparison (s)", "Time (seconds)")
    except Exception as e:
        print(f"(i) Skipped metric plotting: {e}")

if __name__ == "__main__":
    main()