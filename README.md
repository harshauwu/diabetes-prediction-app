# Diabetes Prediction App

This project predicts whether a patient has diabetes using ML models. Built using Python, Pandas, Scikit-learn, and Streamlit.

## Problem Statement
Early detection of diabetes can reduce complications and healthcare costs. This ML solution helps clinicians screen patients using medical data.

## Project Structure
- `data/`: Contains raw and processed datasets
- `notebooks/`: Jupyter notebooks for exploration and model building
- `src/`: Python scripts for preprocessing, training, and prediction
- `app/`: Streamlit app for user interaction
- `models/`: Serialized ML model files
- `tests/`: Unit tests for model functions

## Algorithms Used
- Logistic Regression
- Random Forest
- Evaluation: Accuracy, ROC AUC, Confusion Matrix

## Performance
| Model              | Accuracy | ROC AUC |
|-------------------|----------|---------|
| LogisticRegression| 85%      | 0.91    |
| RandomForest       | 87%      | 0.93    |

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
