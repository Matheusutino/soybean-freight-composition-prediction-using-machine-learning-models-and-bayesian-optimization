from typing import List
import pandas as pd
from src.core.utils import read_lines_csv

class Dataset:
    @staticmethod
    def load_dataset(dataset_path: str) -> pd.DataFrame:
        # lines = read_lines_csv(self.dataset_path)

        # columns = [col.replace("\"", "") for col in lines[0].strip().split(',')]
        # data = [line.strip().replace("\"", "").split(',') for line in lines[1:]]
        # df = pd.DataFrame(data, columns=columns)  

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
            'VOLUME_EXPORTACAO_UF_ORIGEM_ANO', 'PRECOAJUSTADO', 'preco_frete (y)', 
             'PERIODO_SAFRA'
        ]

        df = df[columns]

        return df
    

