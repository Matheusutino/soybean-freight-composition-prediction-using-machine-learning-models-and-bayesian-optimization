import os
import optuna
import statistics
import numpy as np
import pandas as pd
from typing import Literal
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, StratifiedKFold
from src.core.utils import write_json, create_folder
from src.core.metric import Metric
from src.core.encoder import Encoder
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from src.core.models import Model



class BayesianOptimizer:
    def __init__(self, 
                 X: pd.DataFrame, 
                 y: np.ndarray, 
                 task_type: Literal['classification', 'regression'], 
                 model_name: str,
                 n_splits: int = 5,
                 n_repeats: int = 5,
                 seed: int = 42) -> None:
        """
        Initialize the BayesianOptimizer object.

        Args:
            X (pd.DataFrame): Features dataframe.
            y (pd.Series or np.ndarray): Target variable.
            task_type (Literal['classification', 'regression']): Type of machine learning task.
            model_name (str): Name of the model to optimize.
            n_splits (int, optional): Number of folds for cross-validation. Defaults to 5.
            n_repeats (int, optional): Number of repeats for permutation importance. Defaults to 5.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Raises:
            ValueError: If task_type is not 'classification' or 'regression'.
        """
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")

        self.X = X
        self.y = y
        self.task_type = task_type
        self.model_name = model_name
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed

        self.best_X_train_fold = None
        self.best_X_val_fold = None
        self.best_y_train_fold = None
        self.best_y_val_fold = None
        self.best_y_pred_fold = None
        self.best_score = float('-inf') if self.task_type == 'classification' else float('inf')

        # List to store all metrics across trials
        self.all_metrics = []

        self._initialize()
        
    def _initialize_encoder(self) -> None:
        """Initialize the encoder."""
        self.encoder = Encoder(self.model_name, self.task_type)
        self.encoder.fit(self.y)

    def _initialize_cross_validator(self) -> None:
        """Initialize the cross-validator based on the task type."""
        if self.task_type == 'classification':
            self.num_classes = len(np.unique(self.y))
            self.kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        else:
            self.num_classes = 1
            self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

    def _initialize_tf(self) -> None:
        """Initialize TensorFlow settings."""
        tf.random.set_seed(self.seed)
        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=15,         # Stop if val_loss doesn't improve for 15 epochs
            restore_best_weights=True  # Restore model weights from the epoch with the best val_loss
        )

    def _initialize_optuna(self) -> None:
        """Initialize the Optuna study."""
        self.study = optuna.create_study(
            direction='maximize' if self.task_type == 'classification' else 'minimize',
            sampler=TPESampler(seed=self.seed)
        )
        self.metric_to_optimize = 'f1' if self.task_type == 'classification' else 'rmse'

    def _initialize(self) -> None:
        self._initialize_encoder()
        self._initialize_cross_validator()
        self._initialize_tf()
        self._initialize_optuna()

    def train_and_predict(self, X_train, y_train, X_val):
        """Treina o modelo e faz previsões nos dados de validação."""
        if self.model_name == 'MLP':
            self.model.fit(X_train, y_train, validation_split=0.2, epochs=5, callbacks=[self.early_stopping])
        else:
            self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        return y_pred
    
    def evaluate_performance(self, y_val, y_pred):
        """Avalia o desempenho do modelo com base no tipo de tarefa."""
        metric = Metric(y_val, y_pred, task_type = self.task_type, seed = self.seed)
        if self.task_type == 'classification':
            return metric.evaluate_classification()
        else:
            return metric.evaluate_regression()
    
    def compute_folds_metrics(self, fold_scores: dict) -> dict:
        # Calculate mean metrics across folds
        mean_metrics = {}
        std_metrics = {}

        for metric in fold_scores['fold_0'].keys():
            metric_values = [d[metric] for d in fold_scores.values()]
            mean_metrics[metric] = sum(metric_values) / len(metric_values)
            std_metrics[f'{metric}_std'] = statistics.stdev(metric_values)

        combined_dict = {**fold_scores, **mean_metrics, **std_metrics}

        return combined_dict

    def objective(self, trial: optuna.trial) -> float:
        """
        Objective function for Optuna's optimization.

        Args:
            trial (optuna.trial): Trial object for optimization.

        Returns:
            float: Mean value of the metric to optimize across all K folds.
        """
        # Get model and its parameters based on trial
        self.model = Model().get_model(self.model_name,
                                       self.task_type,
                                       trial,
                                       self.seed,
                                       self.X.shape[1],
                                       self.num_classes)
        
        # Dictionary to store scores for each fold
        fold_scores = {}
        
        best_score_fold = float('-inf') if self.task_type == 'classification' else float('inf')

        # Perform cross-validation
        for fold, (train_index, val_index) in enumerate(self.kf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            y_train = self.encoder.transform(y_train)
            y_pred = self.train_and_predict(X_train, y_train, X_val)

            if(self.model_name == 'MLP'):
                y_pred = self.encoder.transform_inverse(y_pred)
            
            if(self.model_name == 'XGBoost'):
                y_val = self.encoder.transform(y_val)

            score = self.evaluate_performance(y_val, y_pred)
            fold_scores[f'fold_{fold}'] = score

            current_fold_score = score[self.metric_to_optimize]
            if (self.task_type == 'classification' and current_fold_score > best_score_fold) or \
               (self.task_type == 'regression' and current_fold_score < best_score_fold):
                best_score_fold = current_fold_score
                best_X_train_fold = X_train
                best_X_val_fold = X_val
                best_y_train_fold = y_train
                best_y_val_fold = y_val
                best_y_pred_fold = y_pred
                
        # Save all folds
        combined_metrics = self.compute_folds_metrics(fold_scores)
        self.all_metrics.append(combined_metrics)

        # Save the best fold
        current_score = combined_metrics[self.metric_to_optimize]
        if (self.task_type == 'classification' and current_score > self.best_score) or \
               (self.task_type == 'regression' and current_score < self.best_score):
            self.best_score = current_score
            self.best_X_train_fold = best_X_train_fold
            self.best_X_val_fold = best_X_val_fold
            self.best_y_train_fold = best_y_train_fold
            self.best_y_val_fold = best_y_val_fold
            self.best_y_pred_fold = best_y_pred_fold

        # Return mean value of the metric to optimize
        return current_score

    def save_best_results_json(self, df: pd.DataFrame, path: str):
        dict_best_model = {
                'model_name': self.model_name,
                'task_type': self.task_type,
            }

        ascending = False if self.task_type == 'classification' else True
        df = df.sort_values(by=self.metric_to_optimize, ascending=ascending)
        best_model_row = df.iloc[0] 

        for col in df.columns[2:]:
            value = best_model_row[col]
            if isinstance(value, (int, float, str, bool, dict)):
                dict_best_model[col] = value
            else:
                dict_best_model[col] = value.item() if hasattr(value, 'item') else str(value)

        write_json(path + '/best_model_infos.json', dict_best_model)

    def get_best_model_infos(self, df: pd.DataFrame, path: str) -> None:
        """
        Extract and save information about the best model found during optimization.

        Args:
            path (str): Path to save the best model information.

        Raises:
            ValueError: If df is empty or does not contain necessary columns.
        """
        try:
            if(self.model_name != 'MLP'):
                self.model.set_params(**self.study.best_params)
                        
                y_pred = self.train_and_predict(self.best_X_train_fold, self.best_y_train_fold, self.best_X_val_fold)

                metric = Metric(self.best_y_val_fold, y_pred, task_type = self.task_type, seed = self.seed)

                metric.plot_feature_importance(model = self.model, 
                                                model_name = self.model_name,
                                                X_val = self.best_X_val_fold,
                                                y_val = self.best_y_val_fold,
                                                top_k_features = 10,
                                                feature_names = self.X.columns,
                                                n_repeats = self.n_repeats,
                                                path = path,
                                                encoder = self.encoder)
            
                if(self.model_name == 'XGBoost'):
                    y_pred = self.encoder.transform_inverse(y_pred)
                    self.best_y_val_fold = self.encoder.transform_inverse(self.best_y_val_fold)
                    metric = Metric(self.best_y_val_fold, y_pred, task_type = self.task_type, seed = self.seed)

                if self.task_type == 'classification':
                    metric.save_confusion_matrix(path)
                    
            self.save_best_results_json(df, path)
        except Exception as e:
            raise ValueError(f"Error in extracting best model information: {str(e)}")
        
    def optimize(self, n_trials: int = 50) -> None:
        """
        Perform optimization using Optuna.

        Args:
            n_trials (int, optional): Number of trials for optimization. Defaults to 50.
        
        Raises:
            RuntimeError: If during program run time error
        """
        try:
            self.study.optimize(self.objective, n_trials=n_trials)

            path_to_save = f'results/{self.task_type}/{self.model_name}'
            create_folder(path_to_save)

            # Obter todos os resultados do estudo como um DataFrame
            df_results = self.study.trials_dataframe()
            df_results = pd.concat([df_results, pd.DataFrame(self.all_metrics)], axis = 1)
            df_results.to_csv(path_to_save + '/optuna_results.csv', index=False)

            self.get_best_model_infos(df_results, path = path_to_save)
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}")
