from src.reporting import generate_pdf_report
from src.tuning import (
    tune_logistic_regression,
    tune_random_forest,
    tune_gradient_boosting,
    tune_XGboost,
    tune_LightGBM
)
from src.modeling import (
    load_processed_data,
    save_model,
    build_metrics_dataframe,
    plot_model_metrics
)


def main():
    # Carregar dados processados
    print("📂 Carregando dados processados...")
    X_train, y_train, X_test, y_test = load_processed_data()

    # Tuning dos modelos (opcional)
    print("\n⚙️ Iniciando tuning de hiperparâmetros...")

    # 1) Logistic Regression
    lr_tuned = tune_logistic_regression(X_train, y_train, X_test, y_test)
    print("✔ Logistic Regression tunado.")

    # 2) Random Forest
    rf_tuned = tune_random_forest(X_train, y_train, X_test, y_test)
    print("✔ Random Forest tunado.")

    # 3) Gradient Boosting
    gb_tuned = tune_gradient_boosting(X_train, y_train, X_test, y_test)
    print("✔ Gradient Boosting tunado.")

    # 4) XGBoost
    xgb_tuned = tune_XGboost(X_train, y_train, X_test, y_test)
    print("✔ XGBoost tunado.")

    # 5) LightGBM
    lgbm_tuned = tune_LightGBM(X_train, y_train, X_test, y_test)
    print("✔ LightGBM tunado.")
    print("✔ Tuning de hiperparâmetros concluído.")

    # Avaliação dos modelos
    print("\n📊 Avaliando modelos...")

    
    lr_tuned_metrics = lr_tuned['final_metrics']
    rf_tuned_metrics = rf_tuned['final_metrics']
    gb_tuned_metrics = gb_tuned['final_metrics']
    xgb_tuned_metrics = xgb_tuned['final_metrics']
    lgbm_tuned_metrics = lgbm_tuned['final_metrics']


    results_tuned = {
    "Logistic Regression (Tuned)": lr_tuned_metrics,
    "Random Forest (Tuned)": rf_tuned_metrics,
    "Gradient Boosting (Tuned)": gb_tuned_metrics,
    "XGBoost (Tuned)": xgb_tuned_metrics,
    "LightGBM (Tuned)": lgbm_tuned_metrics
    }

    # Salvar modelos
    print("\n💾 Salvando modelos...")
    save_model(lr_tuned['best_model'], "models/logistic_regression_tuned.pkl")
    save_model(rf_tuned['best_model'], "models/random_forest_tuned.pkl")
    save_model(gb_tuned['best_model'], "models/gradient_boosting_tuned.pkl")
    save_model(xgb_tuned['best_model'], "models/xgboost_tuned.pkl")
    save_model(lgbm_tuned['best_model'], "models/lightgbm_tuned.pkl")
    

    print("\n✔ Todos os modelos foram treinados e salvos com sucesso em /models/")


    def print_model_comparison(results):
        print("\n📊 Comparação de Modelos:")
        print("Modelo           ROC-AUC    Recall    Precision")
        print("-----------------------------------------------")
        for model_name, metrics in results.items():
            print(f"{model_name:20} {metrics['roc_auc']:.4f}   {metrics['recall']:.4f}   {metrics['precision']:.4f}")

    print_model_comparison(results_tuned)

    # Construir DataFrame de métricas
    df_tuned = build_metrics_dataframe(results_tuned)
    plot_model_metrics(df_tuned)

    # Gerar relatórios PDF
    generate_pdf_report(df_tuned)

if __name__ == "__main__":
    main()
