from src.modeling import (
    load_processed_data,
    train_logistic_regression,
    evaluate_model,
    save_model,
    train_random_forest,
    train_gradient_boosting
)

def main():
    # Carregar dados processados
    print("📂 Carregando dados processados...")
    X_train, y_train, X_test, y_test = load_processed_data()

    # Treinar Logistic Regression
    print("\n🔹 Treinando Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    print("✔ Logistic Regression treinado.")

    # Treinar Random Forest
    print("\n🔹 Treinando Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    print("✔ Random Forest treinado.")

    # Treinar Gradient Boosting
    print("\n🔹 Treinando Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train, y_train)
    print("✔ Gradient Boosting treinado.")

    # Avaliação dos modelos
    print("\n📊 Avaliando modelos...")

    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    gb_metrics = evaluate_model(gb_model, X_test, y_test)

    print("\n📌 Métricas Logistic Regression:", lr_metrics)
    print("📌 Métricas Random Forest:", rf_metrics)
    print("📌 Métricas Gradient Boosting:", gb_metrics)

    # Salvar modelos
    print("\n💾 Salvando modelos...")
    save_model(lr_model, "models/logistic_regression.pkl")
    save_model(rf_model, "models/random_forest.pkl")
    save_model(gb_model, "models/gradient_boosting.pkl")

    print("\n✔ Todos os modelos foram treinados e salvos com sucesso em /models/")

if __name__ == "__main__":
    main()
