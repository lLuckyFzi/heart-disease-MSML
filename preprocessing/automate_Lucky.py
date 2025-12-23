import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def automate_preprocessing(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} is not found")
        return
    
    df = pd.read_csv(input_path)

    OUTLIER_LABEL = "chol"
    df = df[df[OUTLIER_LABEL] <= 500].copy()

    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)

    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    df_encoded[num_features] = scaler.fit_transform(df_encoded[num_features])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_encoded.to_csv(output_path, index=False)
    
    print(f"Data preprocessed tersimpan di: {output_path}")

if __name__ == "__main__":
    RAW_DATA = "heart_disease.csv"
    OUTPUT_DATA = "preprocessing/heart_disease_preprocessing/hear_disease_clean.csv"

    automate_preprocessing(RAW_DATA, OUTPUT_DATA)