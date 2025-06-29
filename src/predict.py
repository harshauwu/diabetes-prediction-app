import joblib

def make_prediction(model_path, input_data):
    model = joblib.load(model_path)
    return model.predict(input_data)
