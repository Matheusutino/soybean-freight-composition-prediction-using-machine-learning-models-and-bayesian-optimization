import argparse
from src.core.dataset import Dataset
from src.core.preprocessing import Preprocessing
from src.core.bayesian_optimizer import BayesianOptimizer
from src.core.utils import create_folder

def main(task_type, model_name, n_trials):
    df = Dataset().load_dataset(dataset_path='dataset/Banco_de_dados_ajustado_BASE.csv')
    X, y = Preprocessing.clean_data(df, task_type=task_type)

    optimizer = BayesianOptimizer(X, y, task_type=task_type, model_name=model_name)
    df_results = optimizer.optimize(n_trials=n_trials)

    path_to_save = 'results/' + task_type + '/' + model_name
    create_folder(path_to_save)
    df_results.to_csv(path_to_save + '/optuna_results.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for Bayesian optimization with parameters.')
    parser.add_argument('--task_type', type=str, choices=['classification', 'regression'],
                        help='Task type: classification or regression')
    parser.add_argument('--model_name', type=str,
                        help='Model name: name of machine learning model')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of attempts for optimization')

    args = parser.parse_args()

    main(args.task_type, args.model_name, args.n_trials)
