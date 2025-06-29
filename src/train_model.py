from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X, y, model_path):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model
