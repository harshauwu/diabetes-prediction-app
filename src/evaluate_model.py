from sklearn.metrics import classification_report, accuracy_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)
