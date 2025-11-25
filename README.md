# 🏦 Credit Risk ML

### Previsão de inadimplência bancária usando Machine Learning

Este projeto desenvolve um modelo para prever a probabilidade de um cliente se tornar **inadimplente** dentro de 2 anos, utilizando o dataset real "Give Me Some Credit" do Kaggle.

---

## 🚀 Tecnologias Utilizadas

* Python 3.10+
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Imbalanced-learn (SMOTE)
* TensorFlow (CPU)
* Jupyter Notebook

---

## 📁 Estrutura do Projeto

### **1. Coleta de Dados**

Dataset: *Credit Card Fraud Detection* (Kaggle)  
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

Alvo (variável target): `Class`  
- 0 = transação normal  
- 1 = transação fraudulenta (positivo)


---

### **2. Análise Exploratória (EDA)**

* Distribuições
* Identificação de outliers
* Missing values
* Correlação
* Variáveis mais importantes para risco

---

### **3. Preparação dos Dados**

* Tratamento de ausentes
* Normalização (StandardScaler)
* Balanceamento com SMOTE
* Train-test split

---

### **4. Machine Learning**

Modelos usados:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost / LightGBM (opcional)

Métricas:

* ROC-AUC
* Recall
* Precision
* Confusion Matrix

---

### **5. Deep Learning (MLP)**

* Rede neural densa
* Early stopping
* Comparação com ML tradicional

---

### 🎯 Objetivo

Construir um pipeline bancário **realista**, focado em:

* Prever clientes inadimplentes
* Criar um modelo interpretável
* Gerar portfólio forte para Data Science

---

# 🔧 Status

Em desenvolvimento — Sprint 1.

---