import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import make_prediction
from flask import Flask, render_template, request
import numpy as np


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            input_data = [float(request.form[f]) for f in [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]]
            prediction = make_prediction("../models/best_random_forest.pkl", [input_data])[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
