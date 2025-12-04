# 🏦 Credit Fraud Detection — Machine Learning

### Previsão de transações bancárias fraudulentas usando aprendizado de máquina

Este projeto implementa um pipeline completo para **detecção de fraudes em cartões de crédito**, utilizando o dataset real *Credit Card Fraud Detection* do Kaggle.
O objetivo é construir um sistema escalável, interpretável e aplicável a cenários reais do setor bancário — passando por EDA, pré-processamento, modelagem, tuning e geração automática de relatórios.

---

# 📊 Dataset

**Fonte:** Kaggle — *Credit Card Fraud Detection*
Link: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### **Características principais**

* 284.807 transações
* Apenas **0,17%** são fraudes
* Features **V1–V28** geradas via PCA
* Alvo:

  * `0` → legítima
  * `1` → fraudulenta

---

# 🔍 Exploratory Data Analysis (EDA)

✔ Fraudes < 1% (dataset extremamente desbalanceado)
✔ PCA cria componentes informativos → V17, V14, V12 se destacam
✔ `Amount` com cauda longa
✔ Outliers mantidos
✔ Gráficos: histogramas, boxplots, countplot, correlação

---

# 🧹 Pré-processamento

Pipeline em **`src/preprocessing.py`**, contendo:

1. Separação X / y
2. Train-test split 80/20 estratificado
3. Normalização (StandardScaler)
4. Balanceamento com SMOTE
5. Salvamento dos arrays processados

Arquivos gerados:

```
data/processed/
  ├── X_train_bal.npy
  ├── X_test.npy
  ├── y_train_bal.npy
  ├── y_test.npy
```

Scaler salvo em `models/scaler.pkl`.

---

# 🤖 Modelagem — Modelos Base

Foram treinados 5 modelos iniciais:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

Com métricas avaliadas em ROC-AUC, Recall, Precision e matriz de confusão.

(Se desejar manter a seção antiga com resultados *antes* do tuning, deixe como está.)

---

# ⚙️ Tuning de Hiperparâmetros (NOVO — Dia 8)

Nesta etapa, cada modelo foi **otimizado individualmente** usando `RandomizedSearchCV`, sempre com foco em:

✔ Maximizar **ROC-AUC**
✔ Manter execução leve para evitar sobreaquecimento
✔ Reduzir busca para estabilidade e performance

Funções de tuning implementadas em:
**`src/tuning.py`**

Modelos tunados:

### **1. Logistic Regression (Tuned)**

* Ajuste de `C`, `penalty`, `solver`
* Resultado:

  * ROC-AUC: **0.9755**
  * Precision: **0.8235**
  * Recall: **0.5714**

→ Modelo mais conservador após tuning (alta precisão, recall menor).

---

### **2. Random Forest (Tuned)**

* Ajuste de número de árvores, profundidade, min_samples, max_features
* Resultado:

  * ROC-AUC: **0.9652**
  * Recall: **0.7959**
  * Precision: **0.8764**

→ Modelo mais equilibrado e robusto.

---

### **3. Gradient Boosting (Tuned)**

* Ajuste de `n_estimators`, `max_depth`, `learning_rate`, `subsample`
* Resultado:

  * ROC-AUC: **0.9129**
  * Precision: **0.7604**
  * Recall: **0.7449**

→ Performance reduziu por conta da amostra reduzida (esperado).

---

### **4. XGBoost (Tuned)**

* Ajuste de `eta`, `subsample`, `colsample_bytree`, `max_depth`
* (Inclui early stopping automático)

Resultado:

* ROC-AUC: **0.9758** *(melhor do tuning)*
* Precision: **0.8947**
* Recall: **0.6939**

→ Melhor modelo em AUC + precisão.

---

### **5. LightGBM (Tuned)**

* Ajuste de `learning_rate`, `n_estimators`, profundidade, leaves
* Resultado:

  * ROC-AUC: **0.5480**

→ Não performou bem com dataset reduzido (comportamento esperado).

---

# 🏆 Comparação — Modelos Tunados

| Modelo                      | ROC-AUC | Recall | Precision |
| --------------------------- | ------- | ------ | --------- |
| Logistic Regression (Tuned) | 0.9755  | 0.5714 | 0.8235    |
| Random Forest (Tuned)       | 0.9652  | 0.7959 | 0.8764    |
| Gradient Boosting (Tuned)   | 0.9129  | 0.7449 | 0.7604    |
| XGBoost (Tuned)             | 0.9758  | 0.6939 | 0.8947    |
| LightGBM (Tuned)            | 0.5480  | 0.1735 | 0.0829    |

## ✔ Conclusões do Tuning (Dia 8)

* **Melhor modelo geral:** XGBoost (Tuned)
* **Melhor modelo equilibrado:** Random Forest (Tuned)
* **Mais conservador (alta precisão):** Logistic Regression (Tuned)
* **Modelo que falhou com amostra reduzida:** LightGBM (Tuned)

---

# 📄 Relatório PDF Automático — (NOVO)

Agora o projeto gera automaticamente:

✔ Tabela completa de métricas
✔ Gráficos dos modelos
✔ Conclusão automática (melhor AUC, recall, precisão)
✔ PDF final em:

```
reports/model_report.pdf
```

Implementado em **`src/reporting.py`**.

---

# 🔮 Próximas Etapas

### 💡 Machine Learning Avançado

* Threshold tuning
* Grid Search / Optuna
* Ensemble (Stacking)

### 🧠 Deep Learning

* MLP
* BatchNorm + Dropout
* Early Stopping

### 🚀 Deploy

* Pipeline de produção
* API com FastAPI
* Endpoint `/predict`
* Versionamento de modelos

---

# ⚙ Tecnologias Utilizadas

* Python 3.10+
* Pandas / NumPy
* Matplotlib / Seaborn
* Scikit-learn
* Imbalanced-Learn
* XGBoost
* LightGBM
* TensorFlow
* Joblib
* Openpyxl
* ReportLab

---

# 📌 Status Atual

### ✔ Concluído

* EDA
* Pipeline completo
* Balanceamento com SMOTE
* Treinamento de 5 modelos
* Tuning de 5 modelos
* Relatório PDF final
* Comparação automatizada

### ➡ Próxima etapa

* Seleção do modelo final
* API com FastAPI
* Threshold tuning

---
