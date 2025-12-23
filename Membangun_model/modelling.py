import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dagshub.init(repo_owner='lLuckyFzi', repo_name='heart-disease-msml', mlflow=True)

def train_model():
    df = pd.read_csv("heart_disease_preprocessing/heart_disease_clean.csv")
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="RandomForest_Baseline"):
        params = {"n_estimators": 100, "random_state": 42}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        plt.figure(figsize=(5,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png") 

        with open("requirements.txt", "w") as f:
            f.write("pandas\nmlflow\ndagshub\nscikit-learn\nmatplotlib\nseaborn")
        mlflow.log_artifact("requirements.txt")

        print(f"Berhasil! Akurasi: {acc}. Cek Dashboard DagsHub Anda!")

if __name__ == "__main__":
    train_model()