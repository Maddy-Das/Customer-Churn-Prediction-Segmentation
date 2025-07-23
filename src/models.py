from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pandas as pd

def train_classification_models(X_train, y_train):
    import time
    import psutil
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Support Vector Machine": SVC(probability=True)
    }

    trained = {}
    metrics = {"Accuracy": {}, "Time": {}, "Memory": {}}
    
    for name, model in models.items():
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss

        model.fit(X_train, y_train)

        mem_after = process.memory_info().rss
        end_time = time.time()

        trained[name] = model
        metrics["Time"][name] = end_time - start_time
        metrics["Memory"][name] = (mem_after - mem_before) / 1024 / 1024  # in MB

    return trained, metrics


def perform_clustering(X_scaled, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    clustered_df = pd.DataFrame(X_scaled, columns=[f'feature_{i}' for i in range(X_scaled.shape[1])])
    clustered_df['cluster'] = clusters
    return clustered_df
