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

* Fraude representa menos de 1% das transações.
* Indica necessidade de reamostragem (SMOTE).

### ✔ Análise das features

* Variáveis PCA (V1–V28) possuem padrões diferentes entre fraudes e não fraudes.
* Variável `Amount` apresenta cauda longa e alta variabilidade.

### ✔ Correlação

* Componentes **V17, V14 e V12** mostram forte relação com a classe fraudulenta.
* Algumas componentes PCA carregam alto poder discriminativo.

### ✔ Outliers

* Presentes, mas esperados em dados transformados por PCA.
* Mantidos no conjunto.

### ✔ Gráficos utilizados

* Histogramas por classe
* Heatmap de correlação
* Countplot das classes
* Boxplots de variáveis importantes

---

## 🧹 Pré-processamento

O pré-processamento foi implementado em `src/preprocessing.py` dentro de um pipeline automatizado.

### ✔ 1. Separação X / y

* `Class` é a variável-alvo.
* Demais colunas são features.

### ✔ 2. Train-test split (80/20)

* Divisão estratificada para preservar proporção de fraudes.

### ✔ 3. Normalização (StandardScaler)

* Ajustado **somente no conjunto de treino**.
* Aplicado no teste para evitar *data leakage*.
* Scaler salvo em:

```
models/scaler.pkl
```

### ✔ 4. Balanceamento com SMOTE

* Aplicado apenas no treino.
* Aumenta a classe minoritária de forma sintética.
* Melhora o aprendizado em datasets desbalanceados.

### ✔ 5. Salvamento dos dados processados

Arquivos gerados:

```
data/processed/
  ├── X_train_bal.npy
  ├── X_test.npy
  ├── y_train_bal.npy
  ├── y_test.npy
```

### ✔ 6. Pipeline completo (`preprocess_pipeline()`)

Fluxo implementado:

1. Carrega os dados
2. Separa features e target
3. Divide treino/teste
4. Escala os dados
5. Aplica SMOTE
6. Salva scaler + arrays
7. Retorna formas para validação

---

## 🤖 Modelagem — Logistic Regression (Etapa finalizada)

O primeiro modelo treinado foi **Regressão Logística**, utilizando os dados pré-processados.

### 📊 Resultados Obtidos

**ROC-AUC:** 0.9709
**Recall:** 0.9183
**Precision:** 0.0579

### 📌 Matriz de Confusão

|            | Previsto 0 | Previsto 1 |
| ---------- | ---------- | ---------- |
| **Real 0** | 55402      | 1462       |
| **Real 1** | 8          | 90         |

### 📝 Interpretação profissional

* **ROC-AUC de 0.97** → excelente capacidade de separação.
* **Recall = 91,8%** → modelo recupera a maioria das fraudes (prioridade do setor).
* **Precisão baixa (5,7%)** → esperado em datasets extremamente desbalanceados.
* **Apenas 8 fraudes não detectadas** → ótimo desempenho para aplicações reais.

---

## 🔮 Próximas Etapas (Dia 5 em diante)

### Machine Learning:

* Random Forest
* Gradient Boosting
* XGBoost / LightGBM

### Deep Learning:

* MLP (rede neural densa)
* Early Stopping
* Comparação com modelos tradicionais

### Relatórios:

* Tabelas comparativas de métricas
* Gráficos de performance
* Seleção de modelo final para produção

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

**Etapa concluída:**
✔ Pré-processamento completo
✔ Treinamento e avaliação do modelo Logistic Regression

**Próxima etapa:**
➡ Treinar modelos avançados (Random Forest, Gradient Boosting)

---
