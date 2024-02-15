from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
import pandas as pd
import DataProcessor as dp

"""
This class is responsible for loading dataset from csv file, and preproccesing it for training the model
"""
class Data:

    def __init__(self, data_path: str, seq_len: int):
        self.data_prep = None
        self.data_path = data_path
        self.data = self._read_data()
        self.seq_len = seq_len
        self.processor = None

    def _read_data(self):
        df = pd.read_csv(self.data_path)

        # Converting in datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp').sort_index()

        print(df.shape)

        return df

    def prepare_data(self):
        self.processor = dp.DataProcessor(seq_lenght=self.seq_len).fit(self.data.values)
        data_trans = self.processor.transform(self.data.values)
        self.data_prep = self.processor.preprocess(data_trans)

        return self.data_prep
