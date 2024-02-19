
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

        #Adding categorical month column
        df['Mese'] = df.index.month_name()
        print(df.shape)

        return df

    def prepare_data(self, num_col: list = None, cat_col: list = None):

        if num_col is None:
            num_col = self.data.columns
        if cat_col is None:
            cat_col = []

        self.processor = dp.DataProcessor(seq_lenght=self.seq_len, num_columns=num_col).fit(self.data[num_col].values, self.data[cat_col].values)
        data_trans_num, data_trans_cat = self.processor.transform(self.data[num_col].values, self.data[cat_col].values)
        data_trans = dp.np.concatenate((data_trans_num, data_trans_cat),axis=1)
        self.data_prep = self.processor.preprocess(data_trans)

        return self.data_prep
