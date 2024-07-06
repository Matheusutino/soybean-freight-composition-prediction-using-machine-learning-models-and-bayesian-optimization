import optuna
import numpy as np
import pandas as pd
from typing import Literal
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from src.core.utils import write_json, create_folder

class BayesianOptimizer:
    def __init__(self, 
                 X, 
                 y, 
                 task_type: Literal['classification', 'regression'], 
                 model_name: str,
                 n_splits: int = 5,
                 seed: int = 42):
        self.X = X
        self.y = y
        self.task_type = task_type
        self.model_name = model_name
        self.n_splits = n_splits
        self.seed = seed
        self.study = optuna.create_study(
            direction='maximize' if task_type == 'classification' else 'minimize',
            sampler=TPESampler(seed=self.seed)
        )
        self.metric_to_optimize = 'f1' if self.task_type == 'classification' else 'rmse'
        self.all_metrics = []


    def objective(self, trial: optuna.trial):
        model, params = self.get_model(trial)
        
        if self.task_type == 'classification':
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        
        fold_scores = {}

        for fold, (train_index, val_index) in enumerate(kf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if self.task_type == 'classification':
                score = self.evaluate_classification(y_val, y_pred)
            else:
                score = self.evaluate_regression(y_val, y_pred)

            fold_scores[f'fold_{fold}'] = score

        mean_metrics = {}
        for metric in fold_scores['fold_0'].keys():
            mean_metrics[metric] = sum(d[metric] for d in fold_scores.values()) / len(fold_scores)

        combined_dict = {**fold_scores, **mean_metrics}
        self.all_metrics.append(combined_dict)

        return mean_metrics[self.metric_to_optimize]

    def get_model(self, trial: optuna.trial):
        if self.model_name == 'KNN':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 500),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['cityblock', 'cosine', 'euclidean']),
            }
            model = KNeighborsClassifier(n_jobs = -1) if self.task_type == 'classification' else KNeighborsRegressor(n_jobs = -1)
        elif self.model_name == 'SVM':
            params = {
                'C': trial.suggest_float('C', 1e-10, 1e10, log = True),
                'gamma': trial.suggest_float('gamma', 1e-10, 1e1, log = True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            }
            model = SVC(random_state=self.seed) if self.task_type == 'classification' else SVR(random_state=self.seed)
        elif self.model_name == 'DecisionTree':
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            }
            model = DecisionTreeClassifier(random_state=self.seed) if self.task_type == 'classification' else DecisionTreeRegressor(random_state=self.seed)
        elif self.model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            }
            model = RandomForestClassifier(random_state = self.seed, n_jobs = -1) if self.task_type == 'classification' else RandomForestRegressor(random_state = self.seed, n_jobs = -1)

        elif self.model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'max_leaves': trial.suggest_int('max_leaves', 0, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                'lambda': trial.suggest_float('lambda', 0.0, 10.0),
                'alpha': trial.suggest_float('alpha', 0.0, 10.0),
                'gamma': trial.suggest_float('gamma', 0.0, 10.0),
            }
            model = XGBClassifier(use_label_encoder=False, eval_metric='rmse', seed = self.seed) if self.task_type == 'classification' else XGBRegressor(seed = self.seed)
        elif self.model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'num_leaves': trial.suggest_int('num_leaves', 2, 31),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 10),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1)
            }
            model = lgb.LGBMClassifier(random_state=self.seed, verbose = -1, n_jobs = -1) if self.task_type == 'classification' else lgb.LGBMRegressor(random_state=self.seed, verbose = -1, n_jobs = -1)

        return model, params

    def evaluate_classification(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def evaluate_regression(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

    def get_best_model_infos(self, df, path):
        ascending = False if self.task_type == 'classification' else True
        df = df.sort_values(by=self.metric_to_optimize, ascending=ascending)
        best_model_row = df.iloc[0] 

        dict_best_model = {
            'model_name': self.model_name,
            'task_type': self.task_type,
        }

        # Adicionando as últimas 4 colunas ao dicionário
        for col in df.columns[2:]:
            value = best_model_row[col]
            if isinstance(value, (int, float, str, bool)):
                dict_best_model[col] = value
            else:
                dict_best_model[col] = value.item() if hasattr(value, 'item') else str(value)

        write_json(path + '/best_model_infos.json', dict_best_model)
        



    def optimize(self, n_trials: int = 50):
        self.study.optimize(self.objective, n_trials=n_trials)

        path_to_save = 'results/' + self.task_type + '/' + self.model_name
        create_folder(path_to_save)

        # Obter todos os resultados do estudo como um DataFrame
        df_results = self.study.trials_dataframe()

        df_results = pd.concat([df_results, pd.DataFrame(self.all_metrics)], axis = 1)
        df_results.to_csv(path_to_save + '/optuna_results.csv')

        self.get_best_model_infos(df_results, path = path_to_save)

        return df_results
