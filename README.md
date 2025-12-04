# 🏦 Credit Fraud Detection — Machine Learning

### Previsão de transações bancárias fraudulentas usando aprendizado de máquina

Este projeto implementa um pipeline completo para **detecção de fraudes em cartões de crédito**, utilizando o dataset real *Credit Card Fraud Detection* do Kaggle.
O objetivo é construir um sistema escalável, interpretável e aplicável a cenários reais do setor bancário.

---

## 📊 Dataset

**Fonte:** Kaggle — *Credit Card Fraud Detection*
**Link:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### **Características principais**

* 284.807 transações
* Apenas **0,17%** são fraudes (*dataset extremamente desbalanceado*)
* Features **V1–V28** foram obtidas via PCA (dados anonimizados)
* Coluna **Class** é o alvo:

  * `0` → transação legítima
  * `1` → transação fraudulenta

---

## 🔍 Exploratory Data Analysis (EDA)

### ✔ Distribuição das classes

Fraudes representam menos de 1%, exigindo técnicas de balanceamento como SMOTE.

### ✔ Análise das Features

* Componentes PCA apresentam padrões diferentes entre fraude e não fraude.
* `Amount` possui alta variabilidade e cauda longa.

### ✔ Correlação

* **V17, V14 e V12** têm maior peso na detecção de fraude.

### ✔ Outliers

* Mantidos — são esperados após transformação PCA.

### ✔ Gráficos utilizados

* Histogramas
* Countplot da variável alvo
* Heatmap de correlação
* Boxplots

---

## 🧹 Pré-processamento

Pipeline implementado em **`src/preprocessing.py`**.

### ✔ Etapas

1. Separação X / y
2. Train-test split estratificado (80/20)
3. Normalização com **StandardScaler**
4. Balanceamento com **SMOTE**
5. Salvamento dos arrays processados

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

Fluxo completo: carregar → separar → dividir → escalar → balancear → salvar.

---

# 🤖 Modelagem

Foram treinados **5 modelos**:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

Treinamento realizado em **`train_model.py`**.

---

# 📌 1. Logistic Regression

### 📊 Resultados

* **ROC-AUC:** 0.9709
* **Recall:** 0.9183
* **Precision:** 0.0579

### 🧩 Matriz de Confusão

| Real \ Previsto | 0     | 1    |
| --------------- | ----- | ---- |
| **0**           | 55402 | 1462 |
| **1**           | 8     | 90   |

### ✔ Interpretação

Alta separação e excelente recall; precisão baixa é esperada em cenários desbalanceados.

---

# 📌 2. Random Forest

### 📊 Resultados

* **ROC-AUC:** 0.9684
* **Recall:** 0.8265
* **Precision:** 0.8709

### 🧩 Matriz de Confusão

| Real \ Previsto | 0     | 1  |
| --------------- | ----- | -- |
| **0**           | 56852 | 12 |
| **1**           | 17    | 81 |

### ✔ Interpretação

Modelo muito preciso, ideal quando se deseja evitar falsos positivos, mas perde algumas fraudes.

---

# 📌 3. Gradient Boosting

### 📊 Resultados

* **ROC-AUC:** 0.9809
* **Recall:** 0.9183
* **Precision:** 0.1133

### 🧩 Matriz de Confusão

| Real \ Previsto | 0     | 1   |
| --------------- | ----- | --- |
| **0**           | 56160 | 704 |
| **1**           | 8     | 90  |

### ✔ Interpretação

Melhor AUC entre todos os modelos; recall muito alto.

---

# 📌 4. XGBoost

### 📊 Resultados

* **ROC-AUC:** 0.9800
* **Recall:** 0.8775
* **Precision:** 0.2409

### 🧩 Matriz de Confusão

| Real \ Previsto | 0     | 1   |
| --------------- | ----- | --- |
| **0**           | 56593 | 271 |
| **1**           | 12    | 86  |

### ✔ Interpretação

Ótimo equilíbrio entre recall e precisão.

---

# 📌 5. LightGBM

### 📊 Resultados

* **ROC-AUC:** 0.9568
* **Recall:** 0.8367
* **Precision:** 0.6259

### 🧩 Matriz de Confusão

| Real \ Previsto | 0     | 1  |
| --------------- | ----- | -- |
| **0**           | 56815 | 49 |
| **1**           | 16    | 82 |

### ✔ Interpretação

Boa precisão; menor AUC comparado aos demais.

---

# 🏆 Comparação Geral

| Modelo              | ROC-AUC | Recall | Precision |
| ------------------- | ------- | ------ | --------- |
| Logistic Regression | 0.9709  | 0.9183 | 0.0579    |
| Random Forest       | 0.9684  | 0.8265 | 0.8709    |
| Gradient Boosting   | 0.9809  | 0.9183 | 0.1133    |
| XGBoost             | 0.9800  | 0.8775 | 0.2409    |
| LightGBM            | 0.9568  | 0.8367 | 0.6259    |

### ✔ Conclusões Profissionais

* **Melhor AUC:** Gradient Boosting
* **Melhor Recall:** Logistic Regression & Gradient Boosting
* **Melhor Precision:** Random Forest

Cada modelo mostra forças diferentes — excelente caso para ensemble.

---

# 🔮 Próximas Etapas

### 🔧 Machine Learning Avançado

* Hiperparametrização (Grid Search / Optuna)
* Ensemble (Votação, Stacking)

### 🤖 Deep Learning

* MLP
* Dropout / BatchNorm
* Early Stopping

### 🏗 Infraestrutura

* Pipeline de produção
* API com FastAPI
* Script de inferência

---

## ⚙ Tecnologias Utilizadas

* Python 3.10+
* Pandas / NumPy
* Matplotlib / Seaborn
* Scikit-learn
* Imbalanced-Learn
* XGBoost / LightGBM
* TensorFlow
* Joblib
* Jupyter Notebook
* ReportLab
* Openpyxl

---

## 📌 Status Atual

### ✔ Concluído

* EDA completo
* Pipeline de pré-processamento
* SMOTE
* Treinamento e comparação de **5 modelos**
* Geração de métricas e gráficos

### ➡ Próxima Etapa

* Tuning
* API
* Modelo final para produção

---