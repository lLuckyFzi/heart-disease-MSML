# For Kriteria 2 (Advanced)

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

if os.getenv("GITHUB_ACTIONS") == "true":
    mlflow.set_tracking_uri("file:./mlruns")
else:
    mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("Heart Disease Tuning")

def train_tuning():
    df = pd.read_csv("heart_disease_preprocessing/heart_disease_clean.csv")
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="Tuning_Model_RF"):
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3
        )
        grid_search.fit(X_train, y_train)

        mlflow.log_params(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        acc = accuracy_score(y_test, best_model.predict(X_test))
        mlflow.log_metric("tuned_accuracy", acc)

        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print(f"Tuning Selesai. Akurasi Terbaik: {acc}")

        plt.figure(figsize=(6,5))
        y_pred = best_model.predict(X_test)
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.title("Confusion Matrix - Tuned Model")
        plt.savefig("confusion_matrix_tuned.png")
        mlflow.log_artifact("confusion_matrix_tuned.png")

        report = classification_report(y_test, y_pred)
        with open("classification_report_tuned.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report_tuned.txt")

if __name__ == "__main__":
    train_tuning()