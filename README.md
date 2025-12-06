# 🏦 Bank Fraud Detection — End-to-End Machine Learning Project

Este projeto implementa um **sistema completo de detecção de fraude em transações financeiras**, cobrindo **todo o ciclo de vida de um modelo de Machine Learning**, desde a exploração dos dados até a disponibilização do modelo em produção via **API REST com FastAPI**.

O foco é demonstrar **boas práticas de Data Science e Machine Learning Engineering**, incluindo treinamento, tuning, avaliação, versionamento de modelos e inferência em tempo real e em lote.

---

## 📌 Objetivo do Projeto

Detectar transações fraudulentas de cartão de crédito a partir de dados históricos anonimizados, utilizando modelos de Machine Learning supervisionados e disponibilizando as previsões por meio de uma API.

---

## 🗂️ Estrutura do Projeto

```
bank-fraud-ml/
│
├── api/                    # API FastAPI e client de consumo
│   ├── app.py
│   ├── client.py
│
├── data/
│   ├── raw/               # Dados brutos
│   └── processed/         # Dados processados (numpy arrays)
│
├── models/                # Modelos treinados, scaler e artefatos
│
├── notebooks/
│   └── 01_data_exploracao.ipynb
│
├── reports/               # Métricas, plots e relatório PDF
│
├── src/                   # Código principal de ML
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── tuning.py
│   ├── inference.py
│   └── reporting.py
│
├── tests/                 # Testes de consumo da API
│   └── test_client.py
│
├── train_model.py         # Pipeline completo de treinamento
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

**Fonte:** Kaggle — *Credit Card Fraud Detection*
Link: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Características

* 284.807 transações
* Apenas **0,17%** são fraudes
* Features **V1–V28** geradas via PCA
* `Class`:

  * `0` → legítima
  * `1` → fraudulenta

---

## ⚙️ Pipeline de Machine Learning

1. **Pré-processamento**

   * Escalonamento das features
   * Separação treino/teste
   * Balanceamento do conjunto de treino

2. **Modelos Treinados**

   * Logistic Regression
   * Random Forest
   * Gradient Boosting
   * XGBoost
   * LightGBM

3. **Tuning de Hiperparâmetros**

   * Ajuste individual por modelo
   * Avaliação com métricas focadas em fraude

4. **Avaliação e Comparação**

   * ROC-AUC
   * Recall
   * Precision
   * Matriz de confusão

5. **Seleção Automática do Melhor Modelo**

   * Score composto:

     * ROC-AUC (50%)
     * Recall (30%)
     * Precision (20%)

6. **Persistência de Artefatos**

   * Modelo final
   * Scaler
   * Ordem das features

---

## 📈 Relatórios

O projeto gera automaticamente:

* Tabela comparativa de métricas (`.csv` e `.xlsx`)
* Gráficos de comparação entre modelos
* Relatório final em **PDF**

📁 Pasta: `reports/`

---

## 🚀 API — FastAPI

A API disponibiliza o modelo final para inferência.

### Iniciar a API

```bash
uvicorn api.app:app --reload
```

📍 Endpoint base:

```
http://127.0.0.1:8000
```

---

### 🔍 Healthcheck

```http
GET /
```

Resposta:

```json
{
  "message": "Credit Fraud Detection API funcionando!"
}
```

---

### 🔮 Previsão Individual

```http
POST /predict
```

Exemplo de requisição:

```json
{
  "Time": 1000,
  "V1": -1.2,
  "V2": 0.4,
  "...": "...",
  "V28": -0.6,
  "Amount": 120.55
}
```

Resposta:

```json
{
  "fraud_probability": 0.91,
  "prediction": 1
}
```

---

### 📦 Previsão em Lote

```http
POST /predict-batch
```

Envia múltiplas transações em uma única requisição.

---

## 🧪 Client Python

O projeto inclui um **client Python** para consumo da API.

Exemplo:

```python
from api.client import FraudClient

client = FraudClient("http://127.0.0.1:8000")

client.healthcheck()
client.predict_single(transaction)
client.predict_batch(transactions)
```

Testes disponíveis em:

```
tests/test_client.py
```

---

## 🛠️ Tecnologias Utilizadas

* Python
* Pandas / NumPy
* Scikit-Learn
* XGBoost
* LightGBM
* FastAPI
* Uvicorn
* Matplotlib / Seaborn
* Joblib

---

## ✅ Principais Diferenciais

* Pipeline **end-to-end**
* Seleção automática do melhor modelo
* API pronta para produção
* Inferência individual e batch
* Client Python para consumo
* Relatórios automatizados
* Estrutura modular e escalável

---

## 🔮 Próximos Passos (Possíveis Extensões)

* Deploy em cloud (Render, AWS, GCP)
* Monitoramento de drift
* Threshold dinâmico para fraude
* Autenticação na API
* Containerização com Docker

---

## 👤 Autor

**Kayke Andrade**
Estudante de Sistemas de Informação
Interesses: Python, Machine Learning, IA e Backend

---