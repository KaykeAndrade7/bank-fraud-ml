import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def subsample_data(X_train, y_train, frac=0.1):
    size = int(len(X_train) * frac)
    return X_train[:size], y_train[:size]


def tune_logistic_regression(X_train, y_train, X_test, y_test):

    X_small, y_small = subsample_data(X_train, y_train)

    model = LogisticRegression(max_iter=300, n_jobs=1)

    param_grid = {
        "C": np.logspace(-3, 2, 6),
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring="roc_auc",
        n_iter=5,
        cv=2,
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    search.fit(X_small, y_small)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    final_metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return {
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "final_metrics": final_metrics,
        "best_model": best_model
    }


def tune_random_forest(X_train, y_train, X_test, y_test):

    X_small, y_small = subsample_data(X_train, y_train)

    model = RandomForestClassifier(
        random_state=42,
        n_jobs=1
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring="roc_auc",
        n_iter=5,
        cv=2,
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    search.fit(X_small, y_small)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    final_metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return {
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "final_metrics": final_metrics,
        "best_model": best_model
    }


def tune_gradient_boosting(X_train, y_train, X_test, y_test):

    X_small, y_small = subsample_data(X_train, y_train)

    model = GradientBoostingClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": np.linspace(0.01, 0.2, 5),
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring="roc_auc",
        n_iter=5,
        cv=2,
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    search.fit(X_small, y_small)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    final_metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return {
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "final_metrics": final_metrics,
        "best_model": best_model
    }


def tune_XGboost(X_train, y_train, X_test, y_test):

    X_small, y_small = subsample_data(X_train, y_train)

    model = XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        nthread=1,
        random_state=42,
        verbosity=0
    )

    param_grid = {
        "n_estimators": [150, 300],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring="roc_auc",
        n_iter=5,
        cv=2,
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    search.fit(X_small, y_small)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    final_metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return {
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "final_metrics": final_metrics,
        "best_model": best_model
    }


def tune_LightGBM(X_train, y_train, X_test, y_test):

    X_small, y_small = subsample_data(X_train, y_train)

    model = LGBMClassifier(
        n_jobs=1,
        device="cpu",
        random_state=42
    )

    param_grid = {
        "n_estimators": [150, 300],
        "learning_rate": [0.01, 0.1],
        "max_depth": [-1, 5, 10],
        "num_leaves": [31, 60],
        "subsample": [0.7, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring="roc_auc",
        n_iter=5,
        cv=2,
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    search.fit(X_small, y_small)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    final_metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return {
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "final_metrics": final_metrics,
        "best_model": best_model
    }
