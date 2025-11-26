# 🏦 Credit Fraud Detection — Machine Learning

### Previsão de transações bancárias fraudulentas usando aprendizado de máquina

Este projeto constrói um pipeline completo para **detectar fraudes em cartões de crédito**, utilizando o dataset real “Credit Card Fraud Detection” do Kaggle.
O foco é desenvolver um modelo robusto, escalável e aplicável a cenários reais de análise bancária.

---

## 📊 Dataset

**Fonte:** Kaggle — *Credit Card Fraud Detection*
Link: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
**Observações principais:**

* 284.807 transações
* Apenas **0.17%** são fraudulentas → *problema severo de desbalanceamento*
* Variáveis V1–V28 foram reduzidas por PCA (dados anonimizados)
* A coluna **Class** é a variável-alvo

  * `0` → transação normal
  * `1` → fraude

---

## 🔍 Exploratory Data Analysis (EDA)

Principais análises realizadas:

### ✔️ Distribuições

* Fraude representa menos de 1% → extremamente desbalanceado.
* Features PCA (V1–V28) apresentam distribuição centrada e comportamento diferente entre fraudes e não fraudes.

### ✔️ Correlação

* Forte correlação negativa entre **V17, V14, V12** e a classe (fraude).
* Isso indica que algumas componentes PCA carregam sinal importante.

### ✔️ Outliers

* Algumas variáveis possuem valores extremos, mas fazem sentido para dados PCA e não foram removidos.

### ✔️ Gráficos utilizados

* Histogramas por classe
* Heatmap de correlação
* Countplot de fraudes
* Boxplots comparativos

---

## 🧹 Pré-processamento

### Passos implementados:

### ✔️ 1. Separação X / y

* `Class` é a coluna alvo
* Todas as demais variáveis → features

### ✔️ 2. Train-test split

* Proporção 80/20
* Estratificado (mantém proporção de fraudes)

### ✔️ 3. Normalização (StandardScaler)

* Aplicado **apenas no treino**
* Transformação posteriormente aplicada ao teste

### ✔️ 4. Balanceamento SMOTE

O SMOTE (*Synthetic Minority Oversampling Technique*) gera novos exemplos sintéticos da classe minoritária.
Aplicamos **somente no conjunto de treino**, evitando vazamento de informação.

### ✔️ 5. Salvamento dos arquivos processados

Todos os arrays são salvos em:

```
data/processed/
  ├── X_train_bal.npy
  ├── X_test.npy
  ├── y_train_bal.npy
  ├── y_test.npy
```

### ✔️ 6. Pipeline completo implementado

Função: **preprocess_pipeline()**

Ela executa:

1. Carregar dados
2. Separar X/y
3. Dividir treino/teste
4. Escalar
5. Balancear
6. Salvar arrays
7. Retornar formas (debug)

---

## 🤖 Modelos (próximas etapas)

Serão implementados:

### Machine Learning

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost / LightGBM

### Métricas

* ROC-AUC
* Recall (prioridade)
* Precision
* Confusion Matrix

### Deep Learning (MLP)

* Rede neural densa
* Early stopping
* Comparação final com modelos clássicos

---

## ⚙️ Tecnologias Utilizadas

* Python 3.10+
* Pandas / NumPy
* Matplotlib / Seaborn
* Scikit-learn
* Imbalanced-Learn (SMOTE)
* TensorFlow (CPU)
* Jupyter Notebook

---

## 📌 Status

**Etapa atual:** Pré-processamento completo finalizado
**Próxima etapa:** Treinamento dos modelos de Machine Learning

---

Se quiser, posso gerar uma **versão ainda mais profissional** com badges e tabela de métricas.
Quer que eu evolua o README nesse estilo?
