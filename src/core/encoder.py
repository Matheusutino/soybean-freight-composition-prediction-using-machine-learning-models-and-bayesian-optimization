import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Encoder:
    def __init__(self, model_name: str, task_type: str) -> None:
        self.model_name = model_name
        self.task_type = task_type

        self.enc = None

    
    def fit(self, y_to_train: np.ndarray) -> None:
        if(self.task_type == 'classification'):
            if(self.model_name == 'XGBoost'):
                self.enc = LabelEncoder()
                self.enc.fit(y_to_train)
            elif(self.model_name == 'MLP'):
                self.enc = OneHotEncoder()
                self.enc.fit(y_to_train.reshape(-1, 1))

    def transform(self, y_to_encoder: np.ndarray) -> np.ndarray:
        y_enc = y_to_encoder
        if(self.task_type == 'classification'):
            if(self.model_name == 'XGBoost'):
                y_enc = self.enc.transform(y_to_encoder)
            elif(self.model_name == 'MLP'):
                y_enc = self.enc.transform(y_to_encoder.reshape(-1, 1)).toarray()
        
        return y_enc 
    
    def transform_inverse(self, y_to_decoder: np.ndarray) -> np.ndarray:
        y_decoded = y_to_decoder
        if(self.task_type == 'classification'):
            if(self.model_name == 'XGBoost'):
                y_decoded = self.enc.inverse_transform(y_to_decoder)
            elif(self.model_name == 'MLP'):
                y_to_decoder = np.eye(y_to_decoder.shape[1])[np.argmax(y_to_decoder, axis=1)].astype(int)
                y_decoded = self.enc.inverse_transform(y_to_decoder).flatten()

        return y_decoded
    
    def get_classes(self):
        return self.enc.classes_
         
