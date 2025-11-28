import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def train_test_split_custom(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def balance_data_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

### Função para salvar os dados processados

''''Esta função cria automaticamente a pasta `data/processed/` e salva os arrays
resultantes do pré-processamento em formato `.npy`. Isso permite que o modelo
de ML carregue os dados rapidamente sem repetir todo o pipeline.'''

def save_processed(X_train_bal, X_test, y_train_bal, y_test, base_path="data/processed/"):
    # Criar a pasta se não existir
    os.makedirs(base_path, exist_ok=True)
    
    # Caminhos dos arquivos
    paths = {
        "X_train_bal": os.path.join(base_path, "X_train_bal.npy"),
        "X_test":      os.path.join(base_path, "X_test.npy"),
        "y_train_bal": os.path.join(base_path, "y_train_bal.npy"),
        "y_test":      os.path.join(base_path, "y_test.npy"),
    }
    
    np.save(paths["X_train_bal"], X_train_bal)
    np.save(paths["X_test"], X_test)
    np.save(paths["y_train_bal"], y_train_bal)
    np.save(paths["y_test"], y_test)

    print("✔ Dados processados salvos com sucesso em:", base_path)
    print("Arquivos salvos:")
    for name, p in paths.items():
        print(f"   - {name}: {p}")

def save_scaler(scaler, path="models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print("✔ Escalador salvo com sucesso em:", path)

def preprocess_pipeline():
    print("🚀 Iniciando pipeline de pré-processamento...")

    # 1 — Carregar os dados (AGORA CERTO)
    df = load_data("data/raw/creditcard.csv")
    print("✔ Dados carregados:", df.shape)

    # 2 — Separar X e y
    X = df.drop("Class", axis=1)
    y = df["Class"]
    print("✔ X e y separados")

    # 3 — Dividir em treino/teste (NOME DA FUNÇÃO CORRIGIDO)
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y)
    print("✔ Dados divididos em treino/teste")

    # 4 — Escalar (NOME DA FUNÇÃO CORRIGIDO)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("✔ Dados escalados")

    # 5 - Salvar o escalador
    save_scaler(scaler)

    # 6 — Balancear com SMOTE
    X_train_bal, y_train_bal = balance_data_smote(X_train_scaled, y_train)
    print("✔ Dados balanceados com SMOTE")

    # 7 — Salvar tudo (CAMINHO RELATIVO)
    save_processed(
        X_train_bal, 
        X_test_scaled, 
        y_train_bal, 
        y_test, 
        base_path="data/processed"
    )

    # 7 — Retornar shapes
    print("\n📐 Shapes finais:")
    print("X_train_bal:", X_train_bal.shape)
    print("y_train_bal:", y_train_bal.shape)
    print("X_test:", X_test_scaled.shape)
    print("y_test:", y_test.shape)

    return {
        "X_train_bal": X_train_bal.shape,
        "y_train_bal": y_train_bal.shape,
        "X_test": X_test_scaled.shape,
        "y_test": y_test.shape
    }

if __name__ == "__main__":
    print("Iniciando pipeline...")
    preprocess_pipeline()