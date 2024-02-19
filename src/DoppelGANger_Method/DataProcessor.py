import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List

"""
According to the paper TimeGAN, this class contains the preprocessing and postprocessing functions for training and
visualizing syntethic data.
"""


class DataProcessor():

    def __init__(self, seq_lenght: int = 24):
        self.scaler = None
        self.seq_lenght = seq_lenght

    def fit(self, data: np.ndarray):
        # Flip the data to make chronological data
        #ori_data = data[::-1]
        # Normalize the data
        self.scaler = MinMaxScaler().fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        # Normalizazzione
        #ori_data = data[::-1]
        ori_data = self.scaler.transform(data)
        return ori_data

    def preprocess(self, data: np.ndarray) -> List[np.ndarray]:
        # Preprocess the dataset
        temp_data = []
        # Cut data by sequence length
        for i in range(0, len(data) - self.seq_lenght):
            _x = data[i:i + self.seq_lenght]
            temp_data.append(_x)

        # Mix the datasets (to make it similar to i.i.d)
        idx = np.random.permutation(len(temp_data))
        preprocessed_data = []
        for i in range(len(temp_data)):
            preprocessed_data.append(temp_data[idx[i]])
        return preprocessed_data

    def reverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(scaled_data)
