import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_basic():
    df = pd.read_csv("heart_disease_preprocessing/heart_disease_clean.csv")
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Baseline_Autolog"):
        params = {"n_estimators": 100, "random_state": 42}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Training Baseline selesai. Akurasi: {acc}")

if __name__ == "__main__":
    train_basic()