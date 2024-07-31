import argparse
from src.core.dataset import Dataset
from src.core.preprocessing import Preprocessing
from src.core.bayesian_optimizer import BayesianOptimizer

def main(task_type: str, model_name: str, n_trials: int, n_splits: int, n_repeats: int) -> None:
    
    """
    Main function to perform Bayesian optimization.

    Args:
        task_type (str): Type of task, 'classification' or 'regression'.
        model_name (str): Name of the machine learning model.
        n_trials (int): Number of attempts for optimization.
        n_splits (int): Number of splits for cross-validation.
        n_repeats (int): Number of repeats for permutation importance.

    Raises:
        Exception: If any other error occurs during the optimization process.
    """
    try:
        df = Dataset().load_dataset(dataset_path='dataset/Banco_de_dados_ajustado_BASE.csv')
        X, y = Preprocessing.clean_data(df, task_type=task_type)

        optimizer = BayesianOptimizer(X, y, task_type=task_type, model_name=model_name, n_splits=n_splits, n_repeats=n_repeats)
        optimizer.optimize(n_trials=n_trials)

    except Exception as e:
        raise Exception(f"An error occurred during optimization: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for Bayesian optimization with parameters.')
    parser.add_argument('--task_type', type=str, choices=['classification', 'regression'],
                        help='Task type: classification or regression')
    parser.add_argument('--model_name', type=str,
                        help='Model name: name of machine learning model')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of attempts for optimization')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='Number of splits for cross-validation')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of repeats for permutation importance')

    args = parser.parse_args()

    main(args.task_type, args.model_name, args.n_trials, args.n_splits, args.n_repeats)

