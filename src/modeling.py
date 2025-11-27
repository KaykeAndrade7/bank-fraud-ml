import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import joblib
from sklearn.metrics import roc_auc_score,recall_score,precision_score,confusion_matrix

def load_processed_data(base_path="data/processed/"):
    os.path.join(base_path)
    X_train = np.load('data/processed/X_train_bal.npy')
    y_train = np.load('data/processed/y_train_bal.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    return X_train, y_train, X_test, y_test

def train_logistic_regression(X_train, y_train):  
    model = LogisticRegression(solver='liblinear', random_state=42, C=1.0)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
    "roc_auc": roc_auc,
    "recall": recall,
    "precision": precision,
    "conf_matrix": cm
    }

def save_model(model, path="models/logistic_regression.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✔ Modelo salvo em: {path}")