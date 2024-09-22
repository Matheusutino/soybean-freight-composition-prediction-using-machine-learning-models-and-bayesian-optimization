from typing import Literal, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


class Preprocessing:
    @staticmethod
    def clean_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            # Seleciona as colunas numéricas e categóricas
            numeric_cols = X_train.select_dtypes(include='number').columns
            categorical_cols = X_train.select_dtypes(include='object').columns

            # Tratamento para colunas numéricas
            knn_imputer = KNNImputer(n_neighbors=5)
            X_train_numeric = pd.DataFrame(knn_imputer.fit_transform(X_train[numeric_cols]), 
                                        columns=numeric_cols, 
                                        index=X_train.index)
            X_test_numeric = pd.DataFrame(knn_imputer.transform(X_test[numeric_cols]), 
                                        columns=numeric_cols, 
                                        index=X_test.index)

            # Padronização dos dados numéricos
            scaler = StandardScaler()
            X_train_numeric = pd.DataFrame(scaler.fit_transform(X_train_numeric), 
                                        columns=numeric_cols, 
                                        index=X_train.index)
            X_test_numeric = pd.DataFrame(scaler.transform(X_test_numeric), 
                                        columns=numeric_cols, 
                                        index=X_test.index)

            # Tratamento para colunas categóricas
            onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_train_categorical = pd.DataFrame(onehot_encoder.fit_transform(X_train[categorical_cols]), 
                                            columns=onehot_encoder.get_feature_names_out(categorical_cols), 
                                            index=X_train.index)
            
            X_test_categorical = pd.DataFrame(onehot_encoder.transform(X_test[categorical_cols]), 
                                            columns=onehot_encoder.get_feature_names_out(categorical_cols), 
                                            index=X_test.index)

            # Combina os dados numéricos e categóricos
            X_train_transformed = pd.concat([X_train_numeric, X_train_categorical], axis=1)
            X_test_transformed = pd.concat([X_test_numeric, X_test_categorical], axis=1)

            return X_train_transformed, X_test_transformed
        
        except Exception as e:
            print(f"Erro ao limpar os dados: {e}")
            return X_train, X_test


