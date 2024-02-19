import sys

import numpy as np
from numpy import ndarray, dtype, void
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import List, Tuple, Any, Union
from tqdm import tqdm

"""
According to the paper TimeGAN, this class contains the preprocessing and postprocessing functions for training and
visualizing syntethic data.
"""


class DataProcessor():

    def __init__(self, seq_lenght: int, num_columns: list):
        self.scaler_num = None
        self.scaler_cat = None
        self.num_columns = num_columns
        self.seq_lenght = seq_lenght

    def fit(self, data: np.ndarray, cat_data: np.ndarray) -> 'DataProcessor':
        # Flip the data to make chronological data
        ori_data = data[::-1]
        ori_cat_data = cat_data[::-1]
        # Normalize the data
        if data and data.shape[0] > 0:
            self.scaler_num = MinMaxScaler().fit(ori_data)
        else:
            raise ValueError("Data is empty")

        if cat_data and cat_data.shape[0] > 0:
            self.scaler_cat = OneHotEncoder().fit(ori_cat_data)

        return self

    def transform(self, data: np.ndarray, cat_data: np.ndarray) -> Tuple[
        Any, ndarray[Any, dtype[Any]]]:
        # Normalizazzione
        ori_data = data[::-1]
        ori_cat_data = cat_data[::-1]
        if self.scaler_num and data.shape[0] > 0:
            ori_data = self.scaler_num.transform(ori_data)
        else:
            raise ValueError("Data is empty")

        if self.scaler_cat and cat_data.shape[0] > 0:
            ori_cat_data = self.scaler_cat.transform(ori_cat_data).toarray()

        return ori_data, ori_cat_data

    def preprocess(self, data: np.ndarray) -> List[np.ndarray]:
        # Preprocess the dataset
        temp_data = []
        # Cut data by sequence length
        for i in tqdm(range(0, len(data) - self.seq_lenght), file=sys.stdout, desc="Preprocessing"):
            _x = data[i:i + self.seq_lenght]
            temp_data.append(_x)

        # Mix the datasets (to make it similar to i.i.d)
        idx = np.random.permutation(len(temp_data))
        preprocessed_data = []
        for i in tqdm(range(len(temp_data)), file=sys.stdout, desc="Mixing"):
            preprocessed_data.append(temp_data[idx[i]])
        return preprocessed_data

    def reverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        if self.scaler_num:
            inverse_num = self.scaler_num.inverse_transform(scaled_data[:, :len(self.num_columns)])
        else:
            raise ValueError("No processor for numerical data")

        if self.scaler_cat:
            inverse_cat = self.scaler_cat.inverse_transform(scaled_data[:, len(self.num_columns):])
        else:
            raise ValueError("No processor for categories")

        return np.concatenate([inverse_num, inverse_cat], axis=1)
