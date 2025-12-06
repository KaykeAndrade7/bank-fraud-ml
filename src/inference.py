import joblib
import numpy as np
import pandas as pd
import json
from src.modeling import select_best_model

# Variáveis globais (carregadas no startup)
model_final = None
scaler = None
feature_order = None


def load_inference_assets():
    """
    Carrega modelo, scaler e ordem das colunas uma única vez (startup da API).
    """
    global model_final, scaler, feature_order

    print("🔄 Carregando modelo e scaler para inferência...")

    model_final = joblib.load("models/modelo_final.pkl")
    scaler = joblib.load("models/scaler.pkl")

    with open("models/feature_order.json", "r") as f:
        feature_order = json.load(f)

    print("✔ Modelo, scaler e colunas carregados com sucesso!")


def predict_pipeline(input_data):
    """
    Pipeline unificado — aceita dict ou DataFrame.
    Usa SEMPRE o modelo carregado em memória.
    """

    # 1 — Converter para DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # 2 — Ordenar colunas na ordem correta
    df = df[feature_order]

    # 3 — Aplicar scaler
    X = scaler.transform(df)

    # 4 — Obter probabilidades (classe 1)
    prob = model_final.predict_proba(X)[:, 1]

    # 5 — Classes finais
    classes = (prob >= 0.5).astype(int)

    # 6 — Formatando a resposta
    return [
        {
            "fraud_probability": float(prob[i]),
            "prediction": int(classes[i])
        }
        for i in range(len(df))
    ]


def predict_single_transaction(data: dict):
    """
    Aceita apenas 1 transação (dict)
    """
    return predict_pipeline(data)[0]


def predict_batch(df: pd.DataFrame):
    """
    Aceita DataFrame com várias linhas
    """
    return predict_pipeline(df)
