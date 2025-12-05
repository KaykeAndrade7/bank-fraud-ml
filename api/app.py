from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.inference import (
    predict_single_transaction,
    predict_batch,
    load_inference_assets
)

app = FastAPI(
    title="Credit Fraud Detection API",
    version="1.0.0",
    description="API para previsão de fraudes em transações bancárias"
)

# Executado automaticamente ao iniciar a API
@app.on_event("startup")
def startup_event():
    load_inference_assets()


# Entrada Pydantic - Todas as features do modelo
class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# Rota base
@app.get("/")
def home():
    return {"message": "Credit Fraud Detection API funcionando!"}


# Previsão individual
@app.post("/predict", tags=["Predictions"])
def predict(transaction: TransactionInput):

    data = transaction.dict()
    resultado = predict_single_transaction(data)

    return {
        "fraud_probability": resultado["fraud_probability"],
        "prediction": resultado["prediction"]
    }


# Previsão em lote
@app.post("/predict-batch", tags=["Predictions"])
def predict_batch_api(transactions: list[TransactionInput]):

    if len(transactions) == 0:
        return {"error": "Lista vazia recebida. Envie pelo menos 1 transação."}

    df = pd.DataFrame([t.dict() for t in transactions])
    resultados = predict_batch(df)

    return {"results": resultados}
