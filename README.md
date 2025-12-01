# 🏦 Credit Fraud Detection — Machine Learning

### Previsão de transações bancárias fraudulentas usando aprendizado de máquina

Este projeto implementa um pipeline completo para **detecção de fraudes em cartões de crédito**, utilizando o dataset real *Credit Card Fraud Detection* do Kaggle.
O foco é construir um sistema escalável, interpretável e aplicável a cenários reais do setor bancário.

---

## 📊 Dataset

**Fonte:** Kaggle — *Credit Card Fraud Detection*
**Link:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### **Características principais:**

* 284.807 transações
* Apenas **0,17%** são fraudulentas → *problema severamente desbalanceado*
* Features V1–V28 geradas por PCA (dados anonimizados)
* Coluna **Class** é a variável-alvo:

  * `0` → transação normal
  * `1` → transação fraudulenta

---

## 🔍 Exploratory Data Analysis (EDA)

### ✔ Distribuição das classes

* Fraudes representam menos de 1%.
* Indica necessidade de técnicas de balanceamento (SMOTE).

### ✔ Análise das Features

* Variáveis PCA apresentam padrões distintos entre fraudes e não fraudes.
* `Amount` apresenta cauda longa e variância elevada.

### ✔ Correlação

* Componentes **V17, V14 e V12** têm forte correlação com a classe.
* PCA preserva sinais importantes para classificação.

### ✔ Outliers

* Mantidos, pois são esperados após transformação PCA.

### ✔ Gráficos utilizados

* Histogramas por classe
* Countplot da variável alvo
* Heatmap de correlação
* Boxplots exploratórios

---

## 🧹 Pré-processamento

Implementado em **`src/preprocessing.py`** como um pipeline automatizado e modular.

### ✔ 1. Separação X / y

* `Class` = target
* Demais colunas = features

### ✔ 2. Train-Test Split (80/20)

* Divisão estratificada para manter a proporção real de fraudes.

### ✔ 3. Normalização (StandardScaler)

* Ajustado **somente no treino**
* Aplicado no teste para evitar *data leakage*
* Scaler salvo em:

```
models/scaler.pkl
```

### ✔ 4. Balanceamento com SMOTE

* Aplicado **apenas no treino**
* Cria exemplos sintéticos da classe minoritária

### ✔ 5. Salvamento dos dados processados

Arquivos gerados em:

```
data/processed/
  ├── X_train_bal.npy
  ├── X_test.npy
  ├── y_train_bal.npy
  ├── y_test.npy
```

### ✔ 6. Pipeline completo (`preprocess_pipeline()`)

Fluxo:

1. Carrega dados
2. Separa features e target
3. Divide treino/teste
4. Escala
5. Aplica SMOTE
6. Salva scaler + arrays
7. Retorna formatos finais

---

# 🤖 Modelagem

Após o pré-processamento, foram treinados três modelos:

---

# **📌 1. Logistic Regression**

### 📊 Resultados

**ROC-AUC:** 0.9709
**Recall:** 0.9183
**Precision:** 0.0579

### 🧩 Matriz de Confusão

|            | Previsto 0 | Previsto 1 |
| ---------- | ---------- | ---------- |
| **Real 0** | 55402      | 1462       |
| **Real 1** | 8          | 90         |

### ✔ Interpretação

* Excelente separação geral (AUC 0.97)
* Ótimo recall (captura a maioria das fraudes)
* Baixa precisão devido ao desbalanceamento
* Erra pouco em deixar fraudes passarem (somente 8)

---

# **📌 2. Random Forest**

### 📊 Resultados

**ROC-AUC:** 0.9684
**Recall:** 0.8265
**Precision:** 0.8709

### 🧩 Matriz de Confusão

|            | Previsto 0 | Previsto 1 |
| ---------- | ---------- | ---------- |
| **Real 0** | 56852      | 12         |
| **Real 1** | 17         | 81         |

### ✔ Interpretação

* Altíssima precisão (87%) → excelente para evitar falsos alarmes
* Recall mais baixo que LR/GB (perde algumas fraudes)
* Ótima escolha quando se quer precisão de alertas

---

# **📌 3. Gradient Boosting**

### 📊 Resultados

**ROC-AUC:** 0.9809
**Recall:** 0.9183
**Precision:** 0.1133

### 🧩 Matriz de Confusão

|            | Previsto 0 | Previsto 1 |
| ---------- | ---------- | ---------- |
| **Real 0** | 56160      | 704        |
| **Real 1** | 8          | 90         |

### ✔ Interpretação

* Melhor AUC entre os modelos
* Recall igual ao da Regressão Logística
* Precisão baixa, mas esperada para problemas severamente desbalanceados

---

# 🏆 Comparação Geral dos Modelos

| Modelo              | ROC-AUC | Recall | Precision |
| ------------------- | ------- | ------ | --------- |
| Logistic Regression | 0.9709  | 0.9183 | 0.0579    |
| Random Forest       | 0.9684  | 0.8265 | 0.8709    |
| Gradient Boosting   | 0.9809  | 0.9183 | 0.1133    |

### ✔ Interpretação Profissional

* **Maior AUC:** Gradient Boosting
* **Maior Recall:** Logistic Regression / Gradient Boosting
* **Maior Precision:** Random Forest (de longe)

Cada modelo tem força diferente → ideal para ensemble no futuro.

---

## 🔮 Próximas Etapas 

### ML Avançado

* XGBoost
* LightGBM
* Ensemble (votação ou stacking)

### Deep Learning

* MLP simples
* Batch Normalization
* Early Stopping

### Infraestrutura

* Scripts automatizados
* Comparação final dos modelos
* Seleção de modelo para produção

---

## ⚙ Tecnologias Utilizadas

* Python 3.10+
* Pandas / NumPy
* Matplotlib / Seaborn
* Scikit-learn
* Imbalanced-Learn (SMOTE)
* TensorFlow (CPU)
* Joblib
* Jupyter Notebook

---

## 📌 Status Atual

### ✔ Concluído até agora:

* EDA completo
* Pipeline de pré-processamento
* Balanceamento com SMOTE
* Treinamento de:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting

### ➡ Próxima etapa:

* Modelos avançados e tuning

---
