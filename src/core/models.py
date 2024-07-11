import os
import optuna
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import lightgbm as lgb
import tensorflow as tf

class Model:
    @staticmethod
    def get_model(model_name:str, 
                  task_type:str, 
                  trial: optuna.trial, 
                  seed: int = 42, 
                  n_features: Union[None, int] = None, 
                  num_classes: Union[None, int] = None):

        if model_name == 'KNN':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 500),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['cityblock', 'cosine', 'euclidean']),
            }
            model = KNeighborsClassifier(**params, n_jobs = -1) if task_type == 'classification' else KNeighborsRegressor(**params, n_jobs = -1)
        elif model_name == 'LogisticRegression':
            params = {
                'C': trial.suggest_float('C', 1e-10, 1e10, log = True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
            }
            model = LogisticRegression(**params, max_iter = 300, solver = 'saga', random_state=seed, n_jobs = -1)
        elif model_name == 'SVM':
            params = {
                'C': trial.suggest_float('C', 1e-10, 1e10, log = True),
                'gamma': trial.suggest_float('gamma', 1e-10, 1e1, log = True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            }
            model = SVC(**params, random_state=seed) if task_type == 'classification' else SVR(**params)
        elif model_name == 'DecisionTree':
            params = {
                'criterion': 'squared_error' if task_type == 'regression' else trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            }
            model = DecisionTreeClassifier(**params, random_state=seed) if task_type == 'classification' else DecisionTreeRegressor(**params, random_state=seed)
        elif model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'criterion': 'squared_error' if task_type == 'regression' else trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            }
            model = RandomForestClassifier(**params, random_state = seed, n_jobs = -1) if task_type == 'classification' else RandomForestRegressor(**params, random_state = seed, n_jobs = -1)

        elif model_name == 'XGBoost':
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
            model = XGBClassifier(**params, seed = seed) if task_type == 'classification' else XGBRegressor(**params, seed = seed)
        elif model_name == 'LightGBM':
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
            model = lgb.LGBMClassifier(**params, random_state=seed, verbose = -1, n_jobs = -1) if task_type == 'classification' else lgb.LGBMRegressor(**params, random_state=seed, verbose = -1, n_jobs = -1)
        elif model_name == 'ExtraTrees':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                'criterion': 'squared_error' if task_type == 'regression' else trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            }
            model = ExtraTreesClassifier(**params, random_state=seed, n_jobs = -1) if task_type == 'classification' else ExtraTreesRegressor(**params, random_state=seed, n_jobs = -1)
        elif model_name == 'MLP':
            # Definindo os hiperparâmetros da MLP
            num_layers = trial.suggest_int('num_layers', 1, 4)
            num_units = trial.suggest_int('num_units', 16, 128, step = 16)
            dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.5, step = 0.05)
            learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log = True)

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=(n_features,)))

            for _ in range(num_layers):
                model.add(tf.keras.layers.Dense(num_units, activation='relu'))
                model.add(tf.keras.layers.Dropout(dropout_rate, seed=seed))
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax' if task_type == 'classification' else 'linear'))

            # Compilando o modelo
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                        loss='categorical_crossentropy' if task_type == 'classification' else 'mean_squared_error',
                        metrics=[tf.keras.metrics.F1Score] if task_type == 'classification' else [tf.keras.metrics.RootMeanSquaredError])
        else:
            raise ValueError(f"Invalid model_name '{model_name}' provided.")

        return model