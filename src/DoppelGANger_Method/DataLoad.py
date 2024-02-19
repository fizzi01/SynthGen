import pandas as pd
import DataProcessor as dp

"""
This class is responsible for loading dataset from csv file, and preproccesing it for training the model
"""


class Data:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._read_data()
        self.processor = None

    def _read_data(self):
        df = pd.read_csv(self.data_path)

        # Converting in datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp').sort_index()

        print(df.shape)

        return df

    def prepare_data(self, num_col: list = None, cat_col: list = None):

        if num_col is None:
            num_col = self.data.columns
        if cat_col is None:
            cat_col = []

        self.data['Mese'] = self.data.index.month_name()

        data_tmp = self.data.reset_index(drop=True)
        self.processor = dp.DataProcessor()
        self.processor.fit(data_tmp[num_col].values)
        transformed = self.processor.transform(data_tmp[num_col].values)
        transformed = pd.DataFrame(transformed, columns=num_col).reset_index(drop=True)
        transformed = pd.concat([transformed,data_tmp[cat_col]], axis=1)

        return transformed
