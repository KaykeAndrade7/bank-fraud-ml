from src.modeling import (load_processed_data,train_logistic_regression,evaluate_model,save_model)

def main():
    # Carregar dados processados
    X_train, y_train, X_test, y_test = load_processed_data()

    # Treinar o modelo de regressão logística
    model = train_logistic_regression(X_train, y_train)
    print("✔ Modelo treinado com sucesso.")

    # Avaliar o modelo
    metrics = evaluate_model(model, X_test, y_test)
    print("📊 Métricas de Avaliação:", metrics)

    # Salvar o modelo treinado
    save_model(model)
    print("modelo treinado e salvo com sucesso em /models/logistic_regression.pkl")
    

if __name__ == "__main__":
    main()
