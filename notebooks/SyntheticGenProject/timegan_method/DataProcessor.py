import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import List, Any
from tqdm import tqdm
from joblib import dump, load

"""
According to the paper TimeGAN, this class contains the preprocessing and postprocessing functions for training and
visualizing syntethic data.
"""


def verifica_pattern(finestra, resolution, soglia_bassa):
    ore_notturne_pre = finestra[:5 * resolution]  # 00 - 4:00
    ore_diurne = finestra[5*resolution:19 * resolution]
    ore_notturne_post = finestra[18 * resolution:]  # ('18:00', '23:00')

    # Bounds cut
    ore_notturne_pre[0] = 0
    ore_notturne_post[-1] = 0

    # Controlla il pattern usando le soglie calcolate precedentemente
    if ore_notturne_pre.max() <= soglia_bassa and ore_notturne_post.max() <= soglia_bassa and ((ore_diurne >= soglia_bassa).any()):
        ore_notturne_pre.loc[ore_notturne_pre < soglia_bassa] = 0
        return True
    return False


def extract_samples(real, synth) -> List[Any]:
    ghi_diurni = real['shortwave_radiation_instant (W/m²)'].between_time('05:00', '19:00')

    # Calcolo delle soglie
    soglia_bassa = ghi_diurni.quantile(0.25)

    # Lista per salvare le finestre di 24 ore che corrispondono al pattern
    finestre_valide = []

    # Numero totale di righe nel dataframe
    total_rows = len(synth)

    h_resolution = int(60 / (real.index[1] - real.index[0]).seconds * 60)
    m_resolution = int(24*60/int((real.index[1] - real.index[0]).seconds / 60))

    # Iterare attraverso il DataFrame a blocchi di 24 righe
    for start in tqdm(range(0, total_rows, m_resolution), file=sys.stdout, desc="Extracting samples"):
        # Verifica che ci siano abbastanza righe per una finestra completa di 24 ore
        if start + m_resolution > total_rows:
            break  # Esce dal ciclo se non ci sono abbastanza righe per una nuova finestra

        # Estrai il blocco di 24 righe
        finestra = synth.iloc[start:start + m_resolution]
        finestra = finestra.reset_index(drop=True)

        if verifica_pattern(finestra=finestra['shortwave_radiation_instant (W/m²)'], resolution=h_resolution,
                            soglia_bassa=soglia_bassa):
            # Fixing mismatch
            finestra["Mese"] = finestra["Mese"].mode()[0]
            finestre_valide.append(finestra)

    return finestre_valide


class DataProcessor():

    def __init__(self, seq_lenght: int = 24, num_columns: list = None, cat_columns: list = None):
        self.scaler_num = None
        self.scaler_cat = None
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.seq_lenght = seq_lenght
        self.col_number = 0

    def fit(self, data: np.ndarray, cat_data: np.ndarray) -> 'DataProcessor':
        # Flip the data to make chronological data
        ori_data = data[::-1]
        ori_cat_data = cat_data[::-1]
        # Normalize the data
        if len(ori_data) > 0:
            self.scaler_num = MinMaxScaler().fit(ori_data)
        else:
            raise ValueError("Data is empty")

        if len(cat_data) > 0:
            self.scaler_cat = OneHotEncoder().fit(ori_cat_data)

        return self

    def transform(self, data: np.ndarray, cat_data: np.ndarray):
        # Normalizazzione
        ori_data = data[::-1]
        ori_cat_data = cat_data[::-1]
        if self.scaler_num and data.shape[0] > 0:
            ori_data = self.scaler_num.transform(ori_data)
        else:
            raise ValueError("Data is empty")

        if self.scaler_cat and cat_data.shape[0] > 0:
            ori_cat_data = self.scaler_cat.transform(ori_cat_data).toarray()

        self.col_number = ori_data.shape[1] + ori_cat_data.shape[1]

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

    def load_metadata(self, path: str) -> 'DataProcessor':
        """
        Load processor metadata
        :param path: Pickle file path
        :return: DataProcessor
        """
        processor = load(path)
        self.scaler_num = processor['scaler']
        self.scaler_cat = processor['cat_scaler']
        self.seq_lenght = processor['sequence_length']
        self.num_columns = processor['measurement_cols_metadata']
        self.cat_columns = processor['categorical_cols_metadata']
        self.col_number = processor['col_number']
        return self

    def save_metadata(self, path: str) -> "DataProcessor":
        """
        Saves processor scaler as file
        :return:
        """
        if self.scaler_num is not None:
            dump({
                "scaler": self.scaler_num,
                "cat_scaler": self.scaler_cat,
                "measurement_cols_metadata": self.num_columns,
                "categorical_cols_metadata": self.cat_columns,
                "sequence_length": self.seq_lenght,
                "col_number": self.col_number
            }, path)
        else:
            raise Exception("Missing scaler.")

        return self

    def get_columns(self):
        """Get data column lis"""
        cols = self.num_columns[:]
        cols.extend(self.cat_columns)
        return cols
