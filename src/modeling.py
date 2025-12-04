import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score,recall_score,precision_score,confusion_matrix

def load_processed_data(base_path="data/processed/"):

    path_xtrain = os.path.join(base_path, "X_train_bal.npy")
    path_ytrain = os.path.join(base_path, "y_train_bal.npy")
    path_xtest  = os.path.join(base_path, "X_test.npy")
    path_ytest  = os.path.join(base_path, "y_test.npy")

    X_train = np.load(path_xtrain)
    y_train = np.load(path_ytrain)
    X_test  = np.load(path_xtest)
    y_test  = np.load(path_ytest)
    
    return X_train, y_train, X_test, y_test

def train_logistic_regression(X_train, y_train):  
    model = LogisticRegression(solver='liblinear', random_state=42, C=1.0)
    model.fit(X_train, y_train)
    
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
    model.fit(X_train, y_train)
    
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def build_metrics_dataframe(results: dict):

    os.makedirs("reports", exist_ok=True)

    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "model": model_name,
            "roc_auc": metrics["roc_auc"],
            "recall": metrics["recall"],
            "precision": metrics["precision"]
        })

    df = pd.DataFrame(rows)

    df.to_excel("reports/model_metrics.xlsx", index=False)

    return df


def plot_model_metrics(df):
    os.makedirs("reports/plots", exist_ok=True)
    metrics = ['roc_auc', 'recall', 'precision']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=df.index, y=df[metric])
        plt.xlabel("Modelo")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} — Comparação entre Modelos")
        plt.tight_layout()
        plt.savefig(f"reports/plots/{metric}_comparison.png")
        plt.close()


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

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✔ Modelo salvo em: {path}")