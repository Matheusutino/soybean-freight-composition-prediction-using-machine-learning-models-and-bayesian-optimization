from typing import Literal, Tuple
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    @staticmethod
    def clean_data(df: pd.DataFrame, 
                   task_type: Literal['classification', 'regression']) -> Tuple[pd.DataFrame]:
        df = df.dropna(subset = ['preco_frete (y)'])

        X = df.drop(['preco_frete (y)', 'PRECOAJUSTADO'], axis = 1)

        # Selecione apenas as colunas numéricas
        numeric_cols = X.select_dtypes(include='number').columns

        # Preencha os valores ausentes nas colunas numéricas com a média
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # Selecione apenas as colunas numéricas
        object_cols = X.select_dtypes(include='object').columns

        X = pd.get_dummies(X, columns=object_cols)

        if(task_type == 'classification'):
            y = df['preco_frete (y)'].to_numpy()
        elif(task_type == 'regression'):
            y = df['PRECOAJUSTADO'].to_numpy()
        else:
            raise ValueError("task_type is 'classification' or 'regression'")

        return X, y
    

