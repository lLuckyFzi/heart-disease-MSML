# For Kriteria 3

import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

if os.getenv("GITHUB_ACTIONS") == "true":
    mlflow.set_tracking_uri("file:./mlruns")
else:
    mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("CI Training")

def train_workflow():
    data_path = "heart_disease_preprocessing/heart_disease_clean.csv"
    df = pd.read_csv(data_path)
    
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Github_Actions_Automated_Run"):
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model_workflow")
        
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
        plt.title("Confusion Matrix - Workflow CI")
        plt.savefig("confusion_matrix_workflow.png")
        mlflow.log_artifact("confusion_matrix_workflow.png")
        
        report = classification_report(y_test, y_pred)
        with open("classification_report_workflow.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report_workflow.txt")

        print(f"Workflow Training Selesai. Akurasi: {acc}")

if __name__ == "__main__":
    train_workflow()