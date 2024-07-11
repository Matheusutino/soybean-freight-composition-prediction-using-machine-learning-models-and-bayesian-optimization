import os
import pandas as pd
from tabulate import tabulate
from src.core.utils import read_json

# Função para processar a pasta de resultados
def process_results_directory(results_dir):
    classification_data = []
    regression_data = []

    # Navegar pela pasta de resultados
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "best_model_infos.json":
                filepath = os.path.join(root, file)
                data = read_json(filepath)

                model_name = data.get('model_name', '')
                task_type = data.get('task_type', '')

                if task_type == 'classification':
                    metrics = {key: data.get(key) for key in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']}
                    metrics['model_name'] = model_name
                    metrics['task_type'] = task_type
                    classification_data.append(metrics)

                elif task_type == 'regression':
                    metrics = {key: data.get(key) for key in ['mse', 'rmse', 'mae', 'mdae', 'r2']}
                    metrics['model_name'] = model_name
                    metrics['task_type'] = task_type
                    regression_data.append(metrics)

    # Criar DataFrames a partir das listas de dados
    df_classification = pd.DataFrame(classification_data)
    df_classification = df_classification.sort_values(by='f1', ascending = False)
    df_regression = pd.DataFrame(regression_data)
    df_regression = df_regression.sort_values(by='rmse', ascending = True)

    return df_classification, df_regression

def main() -> None:
    # Caminho para a pasta de resultados
    results_dir = 'results'

    # Processar a pasta e obter os DataFrames
    df_classification, df_regression = process_results_directory(results_dir)

    # Exibir os DataFrames

    print("\nClassification DataFrame:")
    print(tabulate(df_classification, headers = 'keys', tablefmt = 'psql'))

    print("Regression DataFrame:")
    print(tabulate(df_regression, headers = 'keys', tablefmt = 'psql'))

if __name__ == '__main__':
    main()