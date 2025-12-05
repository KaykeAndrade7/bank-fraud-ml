# 🏦 Credit Fraud Detection — Machine Learning

### Previsão de transações bancárias fraudulentas usando aprendizado de máquina

Este projeto implementa um pipeline completo para **detecção de fraudes em cartões de crédito**, utilizando o dataset real *Credit Card Fraud Detection* do Kaggle.
O objetivo é construir um sistema escalável, interpretável e aplicável a cenários reais do setor bancário — passando por EDA, pré-processamento, modelagem, tuning, relatórios automáticos e agora **infraestrutura inicial de produção**.

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

# 🔍 Exploratory Data Analysis (EDA)

✔ Fraudes < 1%
✔ PCA destaca V17, V14 e V12
✔ `Amount` muito assimétrica
✔ Gráficos incluíram histogramas, boxplots, correlação
✔ Outliers mantidos

---

# 🧹 Pré-processamento

Pipeline implementado em `src/preprocessing.py`:

1. Separação X/y
2. Train-test split estratificado
3. Normalização (StandardScaler)
4. Balanceamento SMOTE
5. Salvamento dos arrays pré-processados

Arquivos gerados:

```
data/processed/
  ├── X_train_bal.npy
  ├── X_test.npy
  ├── y_train_bal.npy
  ├── y_test.npy
```

Scaler salvo em:

```
models/scaler.pkl
```

---

# 🤖 Modelagem — Modelos Base

Modelos inicialmente treinados sem tuning:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

---

# ⚙️ Dia 8 — Tuning de Hiperparâmetros (NOVO)

Cada modelo foi otimizado com RandomizedSearchCV.
Objetivos:

✔ Reduzir custo computacional
✔ Aumentar ROC-AUC
✔ Melhorar recall e precisão sem overfit

Funções em: `src/tuning.py`

---

# 🏆 Resultados — Modelos Tunados

| Modelo                      | ROC-AUC | Recall | Precision |
| --------------------------- | ------- | ------ | --------- |
| Logistic Regression (Tuned) | 0.9755  | 0.5714 | 0.8235    |
| Random Forest (Tuned)       | 0.9652  | 0.7959 | 0.8764    |
| Gradient Boosting (Tuned)   | 0.9129  | 0.7449 | 0.7604    |
| XGBoost (Tuned)             | 0.9758  | 0.6939 | 0.8947    |
| LightGBM (Tuned)            | 0.5480  | 0.1735 | 0.0829    |

### Conclusões do Tuning

* **Melhor modelo geral:** XGBoost (Tuned)
* **Mais equilibrado:** Random Forest (Tuned)
* **Maior precisão:** XGBoost (Tuned)
* **Modelo com pior impacto de amostra reduzida:** LightGBM

---

# 📄 Relatório PDF Automático (NOVO)

Gerado automaticamente pelo código:

```
reports/model_report.pdf
```

Inclui:

✔ Tabela de métricas
✔ Gráficos de ROC-AUC, Recall e Precision
✔ Conclusões automáticas
✔ Melhor modelo destacado

Implementação em: `src/reporting.py`.

---

# 🧠 Dia 9 — Preparação para Produção (NOVO)

Nesta etapa o projeto deixa de ser apenas um pipeline offline e passa a ter **estrutura de produção real**.

## ✔ Seleção automática do modelo final

Criado em `src/modeling.py`:

* Combina AUC, Recall e Precision em um **score composto**
* Retorna automaticamente:

  * nome do melhor modelo
  * caminho do arquivo .pkl
  * score final

O modelo selecionado é salvo como:

```
models/modelo_final.pkl
```

---

# 🧪 Funções de Inferência (NOVO)

Criado o módulo:

```
src/inference.py
```

Contém:

### ✔ `predict_single_transaction()`

Recebe um dicionário → retorna:

* probabilidade de fraude
* classe prevista

### ✔ `predict_batch()`

Recebe um DataFrame → retorna lista de previsões.

### ✔ `predict_pipeline()`

Pipeline real usado em produção:

* carrega scaler e modelo
* ordena features
* aplica normalização
* roda predição
* retorna saída padronizada

---

# 🚀 API com FastAPI (NOVO — Dia 9)

Criada a estrutura inicial em:

```
api/app.py
```

### Endpoints disponíveis:

#### ✔ `GET /`

Teste simples da API.

#### ✔ `POST /predict`

Recebe uma transação
Retorna:

```json
{
  "fraud_probability": 0.87,
  "prediction": 1
}
```

#### ✔ `POST /predict-batch`

Recebe lista de transações
Retorna previsões em lote.

### Carregamento automático

Ao iniciar a API:

✔ modelo_final.pkl
✔ scaler.pkl
✔ feature_order.json

são carregados automaticamente.

---

# 🧩 Estrutura Atualizada do Projeto

```
/api
   ├── app.py
   ├── client.py
/src
   ├── preprocessing.py
   ├── modeling.py
   ├── tuning.py
   ├── inference.py
   ├── reporting.py
/models
   ├── modelo_final.pkl
   ├── scaler.pkl
   ├── feature_order.json
/reports
   ├── model_report.pdf
```

---

# 📌 Status Atual (Atualizado até Dia 9)

### ✔ Concluído

✓ EDA completo
✓ Pré-processamento + SMOTE
✓ Treinamento de 5 modelos
✓ Tuning de 5 modelos
✓ Avaliação comparativa
✓ Gráficos automatizados
✓ Relatório PDF
✓ Seleção automática do melhor modelo
✓ Criação completa da API (predict e batch)
✓ Pipeline real de inferência
✓ Modelo final salvo

---

# 🔮 Próximas Etapas 

* Ajuste fino do threshold
* Stacking/Ensemble avançado
* Persistência de logs
* Deploy na nuvem (Railway / Render / AWS)
* Monitoramento de drift
* Interface web simples (Streamlit)
* Dockerização

---

