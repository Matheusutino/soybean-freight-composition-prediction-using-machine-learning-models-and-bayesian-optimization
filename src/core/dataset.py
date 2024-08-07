import pandas as pd

class Dataset:
    @staticmethod
    def load_dataset(dataset_path: str) -> pd.DataFrame:
        """
        Load a dataset from a CSV file into a Pandas DataFrame.

        Args:
            dataset_path (str): The file path to the CSV dataset.

        Returns:
            pd.DataFrame: The loaded dataset as a Pandas DataFrame.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            pd.errors.EmptyDataError: If the CSV file is empty or cannot be parsed.
            KeyError: If any of the required columns are missing after selection.
        """
        try:
            df = pd.read_csv(dataset_path)

            columns = [
                'ORIGEM', 'UF_ORIGEM', 'DESTINO', 'UF_DESTINO', 
                'DISTANCIA', 'MES_NUMERICO', 'ANO', 'PRODUCAO_MUNI', 'VALOR_PROD_MUNI', 
                'REDIMENTO_MEDIO_MUNI', 'AREA_PLANTADA_MUNI', 'AREA_COLHIDA_MUNI', 'PRODUCAO_ESTADO', 'RENDIMENTO_ESTADO', 
                'AREA_PLANTADA_ESTADO', 'AREA_COLHIDA_ESTADO', 'AREA PLANTADA_ESTADO = AREA COLHIDA_ESTADO', 
                'DIESEL_PRECO_MIN', 'DIESEL_PRECO_MED', 'DIESEL_PRECO_MAX', 
                'ETANOL_PRECO_MIN', 'ETANOL_PRECO_MED', 'ETANOL_PRECO_MAX', 
                'GASOLINA_PRECO_MIN', 'GASOLINA_PRECO_MED', 'GASOLINA_PRECO_MAX', 
                'CAPACIDADE_ARMAZENAGEM_ESTADUAL_ORIGEM', 'CAPACIDADE_ARMAZENAGEM_ESTADUAL_DESTINO', 
                'MERCADO_NACIONAL', 'MERCADO_INTERNACIONAL_CHICAGO', 'MERCADO_INTERNACIONAL_PARIDADE', 
                'CAPACIDADE_INDUSTRIA_ESTADO_ORIGEM', 'CAPACIDADE_INDUSTRIA_ESTADO_DESTINO', 
                'CAMBIO_MEDIO_MENSAL', 'IMPORTACAO_OLEO_DIESEL', 'VOLUME_EXPORTACAO_UF_ORIGEM_MES', 
                'VOLUME_EXPORTACAO_UF_ORIGEM_ANO', 'PRECOAJUSTADO', 'PERIODO_SAFRA'
            ]
            df = df[columns]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {dataset_path}")
        except pd.errors.EmptyDataError as e:
            raise pd.errors.EmptyDataError(f"Empty or unreadable CSV file: {dataset_path}")
        except KeyError as e:
            raise KeyError(f"Required column not found in DataFrame: {e}")

        return df
    

