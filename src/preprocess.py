import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler
