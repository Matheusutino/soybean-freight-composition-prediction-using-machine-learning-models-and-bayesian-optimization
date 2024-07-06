from typing import Literal, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    @staticmethod
    def clean_data(df: pd.DataFrame, 
                   task_type: Literal['classification', 'regression']) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Perform data cleaning and preprocessing on a DataFrame for machine learning tasks.

        Args:
            df (pd.DataFrame): The input DataFrame containing the dataset.
            task_type (Literal['classification', 'regression']): The type of machine learning task.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: A tuple containing the preprocessed features (X) and the target variable (y).

        Raises:
            KeyError: If required columns 'preco_frete (y)' or 'PRECOAJUSTADO' are missing in the DataFrame.
            ValueError: If task_type is neither 'classification' nor 'regression'.
        """
        try:
            # Drop rows with missing values in the target variable column
            df = df.dropna(subset = ['preco_frete (y)'])

            # Drop targets variables
            X = df.drop(['preco_frete (y)', 'PRECOAJUSTADO'], axis = 1)

            # Select only numeric columns
            numeric_cols = X.select_dtypes(include='number').columns

            # Fill missing values in numeric columns by mean, grouped by 'ORIGEM' and 'UF_ORIGEM'
            columns_to_fill_by_origem = ['PRODUCAO_MUNI', 'VALOR_PROD_MUNI', 'REDIMENTO_MEDIO_MUNI', 'AREA_PLANTADA_MUNI', 'AREA_COLHIDA_MUNI']
            columns_to_fill_by_estado_origem = ['CAPACIDADE_INDUSTRIA_ESTADO_ORIGEM', 'VOLUME_EXPORTACAO_UF_ORIGEM_MES']

            X[columns_to_fill_by_origem] = X.groupby('ORIGEM')[columns_to_fill_by_origem].transform(lambda x: x.fillna(x.mean()))
            X[columns_to_fill_by_estado_origem] = X.groupby('UF_ORIGEM')[columns_to_fill_by_estado_origem].transform(lambda x: x.fillna(x.mean()))
            X['CAPACIDADE_INDUSTRIA_ESTADO_DESTINO'] = X.groupby('UF_DESTINO')['CAPACIDADE_INDUSTRIA_ESTADO_DESTINO'].transform(lambda x: x.fillna(x.mean()))

            # Fill any remaining missing numeric values with column means
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

            # Standardize numeric columns using StandardScaler
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

            # Convert object (categorical) columns into dummy variables
            object_cols = X.select_dtypes(include='object').columns
            X = pd.get_dummies(X, columns=object_cols)

            # Select target variable based on task_type
            if(task_type == 'classification'):
                y = df['preco_frete (y)'].to_numpy()
            elif(task_type == 'regression'):
                y = df['PRECOAJUSTADO'].to_numpy()
            else:
                raise ValueError("task_type is 'classification' or 'regression'")

            return X, y
        
        except KeyError as e:
            raise KeyError(f"Required column not found in DataFrame: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid task_type provided: {e}")
    

