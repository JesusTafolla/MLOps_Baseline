import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
from mlflow.models import infer_signature


import mlflow
import mlflow.sklearn

def load_data(filepath):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Limpia los datos y aplica One-Hot Encoding."""
    # Eliminar identificadores irrelevantes
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Corregir TotalCharges a numérico y eliminar nulos
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Codificar variables categóricas
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    X = df_encoded.drop('Churn_Yes', axis=1)
    y = df_encoded['Churn_Yes']
    
    # Evitar problemas de schema con columnas enteras en MLflow
    X = X.astype("float64")

    return X, y

def create_artifacts(model, X_test, y_test, y_pred, y_prob, feature_names):
    """Genera y guarda gráficos como artefactos para MLFlow."""
    os.makedirs("artifacts", exist_ok=True)
    
    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión (Test)")
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC (Test)')
    plt.legend(loc="lower right")
    plt.savefig("artifacts/roc_curve.png")
    plt.close()

    # Feature Importance (Específico para Random Forest)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:] # Top 10 características
    plt.figure(figsize=(8, 6))
    plt.title("Top 10 - Importancia de Características")
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importancia Relativa')
    plt.tight_layout()
    plt.savefig("artifacts/feature_importance.png")
    plt.close()

def main():
    # Configurar Argparse
    parser = argparse.ArgumentParser(
        description="Script de entrenamiento Random Forest para predecir Churn de clientes.",
        epilog="Ejemplo de uso: python train.py --n-estimators 200 --max-depth 8"
    )
    parser.add_argument("--n-estimators", type=int, default=100, 
                        help="Número de árboles (default: 100). Más árboles = mayor estabilidad.")
    parser.add_argument("--max-depth", type=int, default=10, 
                        help="Profundidad del árbol (default: 10). Controla el sobreajuste.")
    parser.add_argument("--min-samples-split", type=int, default=2, 
                        help="Muestras mínimas para dividir (default: 2).")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Semilla aleatoria para reproducibilidad (default: 42).")
    args = parser.parse_args()

    # Carga y preprocesamiento
    print("Cargando y preprocesando datos...")
    df = load_data('telco_churn.csv')
    X, y = preprocess_data(df)
    
    # División estratificada (Train, Val, Test)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=args.seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=args.seed, stratify=y_temp)
    
    print(f"Train: {len(X_train)} registros | Validation: {len(X_val)} registros | Test: {len(X_test)} registros")

    # Iniciar MLFlow Tracking
    mlflow.set_experiment("Telco_Churn_MLOps_Experiment")
    
    with mlflow.start_run(run_name=f"RF_Est_{args.n_estimators}_Depth_{args.max_depth}"):
        print(f"\nEntrenando modelo Random Forest con n_estimators={args.n_estimators}, max_depth={args.max_depth}, min_samples_split={args.min_samples_split}, seed={args.seed}")
        
        # Registrar hiperparámetros
        mlflow.log_params(vars(args))
        
        # Entrenar
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            min_samples_split=args.min_samples_split,
            random_state=args.seed
        )
        model.fit(X_train, y_train)
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Predicciones y métricas en VALIDATION
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_roc_auc = roc_auc_score(y_val, y_val_prob)
        
        # Predicciones y métricas en TEST
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_prob)
        test_recall = recall_score(y_test, y_test_pred)
        
        # Registrar métricas
        metrics = {
            "val_accuracy": val_accuracy,
            "val_roc_auc": val_roc_auc,
            "test_accuracy": test_accuracy,
            "test_roc_auc": test_roc_auc,
            "test_recall": test_recall
        }
        mlflow.log_metrics(metrics)
        
        print("\nResultados de la Corrida:")
        print(f"Validation -> Accuracy: {val_accuracy:.4f} | ROC AUC: {val_roc_auc:.4f}")
        print(f"Test       -> Accuracy: {test_accuracy:.4f} | ROC AUC: {test_roc_auc:.4f} | Recall: {test_recall:.4f}")
        
        # Crear y registrar artefactos
        create_artifacts(model, X_test, y_test, y_test_pred, y_test_prob, X.columns)
        mlflow.log_artifacts("artifacts", artifact_path="eval_plots")
        
        # Registrar el modelo
        mlflow.sklearn.log_model(
            model,
            artifact_path="rf_model",
            signature=signature,
            input_example=X_train.head(1)
        )

        print("\nCorrida finalizada y registrada exitosamente en MLFlow")

if __name__ == "__main__":
    main()