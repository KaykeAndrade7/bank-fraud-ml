from src.modeling import (
    load_processed_data,
    train_logistic_regression,
    evaluate_model,
    save_model,
    train_random_forest,
    train_gradient_boosting,
    train_lightgbm,
    train_xgboost
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

    # Treinar XGBoost
    print("\n🔹 Treinando XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    print("✔ XGBoost treinado.")

    # Treinar LightGBM
    print("\n🔹 Treinando LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train)
    print("✔ LightGBM treinado.")

    # Avaliação dos modelos
    print("\n📊 Avaliando modelos...")

    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    gb_metrics = evaluate_model(gb_model, X_test, y_test)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test)

    print("\n📌 Métricas Logistic Regression:", lr_metrics)
    print("📌 Métricas Random Forest:", rf_metrics)
    print("📌 Métricas Gradient Boosting:", gb_metrics)
    print("📌 Métricas XGBoost:", xgb_metrics)
    print("📌 Métricas LightGBM:", lgbm_metrics)

    results = {
    "Logistic Regression": lr_metrics,
    "Random Forest": rf_metrics,
    "Gradient Boosting": gb_metrics,
    "XGBoost": xgb_metrics,
    "LightGBM": lgbm_metrics
    }

    # Salvar modelos
    print("\n💾 Salvando modelos...")
    save_model(lr_model, "models/logistic_regression.pkl")
    save_model(rf_model, "models/random_forest.pkl")
    save_model(gb_model, "models/gradient_boosting.pkl")
    save_model(xgb_model, "models/xgboost.pkl")
    save_model(lgbm_model, "models/lightgbm.pkl")

    print("\n✔ Todos os modelos foram treinados e salvos com sucesso em /models/")


    def print_model_comparison(results):
        print("\n📊 Comparação de Modelos:")
        print("Modelo               ROC-AUC    Recall    Precision")
        print("-----------------------------------------------")
        for model_name, metrics in results.items():
            print(f"{model_name:20} {metrics['roc_auc']:.4f}   {metrics['recall']:.4f}   {metrics['precision']:.4f}")
    
    print_model_comparison(results)

if __name__ == "__main__":
    main()
