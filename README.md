# 🏦 Credit Fraud Detection — Machine Learning

### Previsão de transações bancárias fraudulentas usando aprendizado de máquina

Este projeto implementa um pipeline completo para **detecção de fraudes em cartões de crédito**, utilizando o dataset real *Credit Card Fraud Detection* do Kaggle.
O objetivo é construir um sistema escalável, interpretável e aplicável a cenários reais do setor bancário.

---

## 📊 Dataset

**Fonte:** Kaggle — *Credit Card Fraud Detection*
**Link:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### **Características principais:**

* 284.807 transações
* Apenas **0,17%** são fraudes (extremamente desbalanceado)
* Features V1–V28 são componentes PCA (dados anonimizados)
* Coluna **Class** é o alvo:

  * `0` → legítima
  * `1` → fraude

---

## 🔍 Exploratory Data Analysis (EDA)

### ✔ Distribuição das classes

* Fraudes < 1% → necessidade de técnicas de balanceamento (SMOTE).

### ✔ Análise das Features

* Features PCA apresentam padrões distintos entre fraudes e não fraudes.
* `Amount` apresenta alta variabilidade e cauda longa.

### ✔ Correlação

* V17, V14 e V12 correlacionam fortemente com a classe.
* PCA preserva componentes discriminativas importantes.

### ✔ Outliers

* Mantidos (esperados após PCA).

### ✔ Gráficos utilizados

* Histogramas
* Countplot
* Heatmap de correlação
* Boxplots

---

## 🧹 Pré-processamento

Pipeline implementado em **`src/preprocessing.py`**.

### ✔ 1. Separação X / y

### ✔ 2. Train-test split (80/20, estratificado)

### ✔ 3. Normalização (StandardScaler)

Scaler salvo em:

```
models/scaler.pkl
```

### ✔ 4. Balanceamento com SMOTE

### ✔ 5. Salvamento dos arrays processados

Arquivos gerados:

```
data/processed/
  ├── X_train_bal.npy
  ├── X_test.npy
  ├── y_train_bal.npy
  ├── y_test.npy
```

### ✔ 6. Pipeline final

Carrega dados → separa → divide → escala → balanceia → salva → retorna shapes.

---

# 🤖 Modelagem

Foram treinados **5 modelos**:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

Todos treinados em `train_model.py`.

---

# 📌 1. Logistic Regression

### 📊 Resultados

* **ROC-AUC:** 0.9709
* **Recall:** 0.9183
* **Precision:** 0.0579

### 🧩 Matriz de Confusão

|            | Prev. 0 | Prev. 1 |
| ---------- | ------- | ------- |
| **Real 0** | 55402   | 1462    |
| **Real 1** | 8       | 90      |

### ✔ Interpretação

* Ótima separação (AUC 0.97)
* Excelente recall
* Baixa precisão, esperado no desbalanceamento

---

# 📌 2. Random Forest

### 📊 Resultados

* **ROC-AUC:** 0.9684
* **Recall:** 0.8265
* **Precision:** 0.8709

### 🧩 Matriz de Confusão

|            | Prev. 0 | Prev. 1 |
| ---------- | ------- | ------- |
| **Real 0** | 56852   | 12      |
| **Real 1** | 17      | 81      |

### ✔ Interpretação

* Altíssima precisão
* Recall mais baixo
* Ideal quando se quer evitar falsos positivos

---

# 📌 3. Gradient Boosting

### 📊 Resultados

* **ROC-AUC:** 0.9809
* **Recall:** 0.9183
* **Precision:** 0.1133

### 🧩 Matriz de Confusão

|            | Prev. 0 | Prev. 1 |
| ---------- | ------- | ------- |
| **Real 0** | 56160   | 704     |
| **Real 1** | 8       | 90      |

### ✔ Interpretação

* Melhor AUC entre os modelos
* Recall excelente
* Precisão baixa devido ao desbalanceamento

---

# 📌 4. XGBoost

### 📊 Resultados

* **ROC-AUC:** 0.9800
* **Recall:** 0.8775
* **Precision:** 0.2409

### 🧩 Matriz de Confusão

|            | Prev. 0 | Prev. 1 |
| ---------- | ------- | ------- |
| **Real 0** | 56593   | 271     |
| **Real 1** | 12      | 86      |

### ✔ Interpretação

* Excelente AUC
* Bom recall
* Melhor precisão que LR/GB

---

# 📌 5. LightGBM

### 📊 Resultados

* **ROC-AUC:** 0.9568
* **Recall:** 0.8367
* **Precision:** 0.6259

### 🧩 Matriz de Confusão

|            | Prev. 0 | Prev. 1 |
| ---------- | ------- | ------- |
| **Real 0** | 56815   | 49      |
| **Real 1** | 16      | 82      |

### ✔ Interpretação

* Excelente precisão
* Bom recall
* Menor AUC que XGBoost/GB

---

# 🏆 Comparação Geral dos Modelos

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
* **Melhor Precision:** Random Forest (de longe)

Cada modelo apresenta vantagens específicas → perfeito para testes de ensemble no futuro.

---

# 🔮 Próximas Etapas

### 🔧 Machine Learning Avançado

* Hiperparametrização (Grid Search / Optuna)
* Ensemble (Votação, Stacking)

### 🤖 Deep Learning

* MLP
* Early Stopping

### 🏗 Infraestrutura

* Pipeline de produção
* FastAPI para servir o modelo
* Script de inferência

---

## ⚙ Tecnologias Utilizadas

* Python 3.10+
* Pandas / NumPy
* Matplotlib / Seaborn
* Scikit-learn
* Imbalanced-Learn
* XGBoost
* LightGBM
* TensorFlow
* Joblib
* Jupyter Notebook

---

## 📌 Status Atual

### ✔ Concluído

* EDA completo
* Pipeline de pré-processamento
* SMOTE
* Treinamento de **5 modelos**
* Comparação completa

### ➡ Próxima Etapa

* Tuning + API
* Escolha do modelo final para produção

---
