import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import make_prediction
from flask import Flask, render_template, request
import numpy as np

import logging
import os

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Configure logging to write to a file
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_values = None

    if request.method == "POST":
        try:
            fields = [
                "pregnancies", "glucose", "blood_pressure", "skin_thickness",
                "insulin", "bmi", "pedigree_function", "age"
            ]
            input_values = {f: float(request.form[f]) for f in fields}
            input_data = [list(input_values.values())]

            prediction = make_prediction("../models/best_random_forest.pkl", [input_data])[0]


            # ✅ Log prediction
            logging.info(f"Prediction input: {input_data}")
            logging.info(f"Prediction result: {prediction}")

        except Exception as e:
            prediction = f"Error: {e}"
            logging.error(f"Prediction error: {e}")

    return render_template("index.html", prediction=prediction, input_values=input_values)



if __name__ == "__main__":
    app.run(debug=True)
