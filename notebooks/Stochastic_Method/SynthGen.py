import math
import sys

import numpy as np
from pvlib import clearsky
from pvlib.location import Location

from tqdm import trange

from DataLoader import *
from Utils import *


class Generator:

    def __init__(self, years: int | list, month: int = 1, resolution: int = 60, file_path: str = '',
                 radiation_col: str = "shortwave_radiation_instant (W/m²)", location: Location = None):

        self.year = years
        self.m = month

        self.ghi_col_name = radiation_col
        self.location = location

        self.res = resolution
        self.file_path = file_path
        self.data_load = DataLoader(filename=self.file_path)

        self._extracted_param: dict = {}
        self.labelled_days: dict = {}

        self._param_matrix: dict = {}
        self._categories: list = []
        self._distr: np.array = None

    def get_days_matrix(self):
        """
        :return: Days values per category
        """
        return self._param_matrix.copy()

    def _extract_data(self):
        """Extract every parameter from data"""
        extracted_param = {}
        for i in trange(len(self.data_load.data.columns), file=sys.stdout, desc='Exctracting data'):
            col = self.data_load.data.columns[i]
            if i == 0 and col == 'timestamp':
                continue
            extracted_param[col] = self.data_load.extract_parameter(parameter=col, year=self.year, month=self.m,
                                                                    resolution=self.res)
        return extracted_param

    def _day_labelling(self) -> dict[Category, list[str]]:
        """
        Categorizza ogni giorno del mese
        :return: dizionario con i giorni divisi per categoria
        """
        tus = self.location

        # Preparo gli indici temporali per i quali ottenere valori di clear-sky
        if isinstance(self.year, list):
            time_ranges = []
            df_cs = []
            for y in self.year:
                start_range = pd.Timestamp(f'{y}-{self.m}-01')
                end_range = pd.Timestamp(start_range) + pd.offsets.MonthEnd(n=1)
                times_tmp = pd.date_range(start=f'{y}-{self.m}-01', end=f'{y}-{self.m}-{end_range.day}',
                                          freq=f'{self.res}min')
                time_ranges.append(times_tmp)
                cs = tus.get_clearsky(times_tmp)  # ineichen with climatology table by default
                df_cs.append(cs)

            cs = pd.concat(df_cs)
            times = pd.DatetimeIndex(np.concatenate(time_ranges))

        else:
            start_range = pd.Timestamp(f'{self.year}-{self.m}-01')
            end_range = pd.Timestamp(start_range) + pd.offsets.MonthEnd(n=1)
            times = pd.date_range(start=f'{self.year}-{self.m}-01', end=f'{self.year}-{self.m}-{end_range.day}',
                                  freq=f'{self.res}min')
            cs = tus.get_clearsky(times)  # ineichen with climatology table by default

        # Recupero la lista dei valori di clear sky da usare come costanti
        constants = {}
        for i, time in enumerate(times):
            constants[time] = cs.iat[i, 0]

        # Categorizzo i giorni in base alla nuvolosità
        return day_categorizer(self._extracted_param[self.ghi_col_name], constants)

    def _params_matrix(self):
        """Per ogni categoria, inserisce al corrispettivo giorno i valori del parametro"""
        tmp_matrix = {}

        for i in trange(len(list(self._extracted_param.items())), file=sys.stdout, desc='Mapping data'):
            param, data = list(self._extracted_param.items())[i]
            tmp_matrix[param] = self._category_matrix(data, self.labelled_days)

        return tmp_matrix

    def generate(self, series_len: int = 1) -> tuple[list[pd.DataFrame], np.array(Category)]:

        self._extracted_param = self._extract_data()
        self.labelled_days = self._day_labelling()

        self._param_matrix = self._params_matrix()
        self._categories, self._distr = self._day_distribution(self.labelled_days)

        days_serie = np.random.choice(a=self._categories, size=series_len, p=self._distr).tolist()

        samples = []
        for i in trange(series_len, file=sys.stdout, desc='Generating samples'):
            day_cat = days_serie[i]
            tmp = []
            for param, matrix in self._param_matrix.items():
                sample = self._generate_synt_data(matrix)
                old_name = sample[day_cat].columns[0]
                sample[day_cat] = sample[day_cat].rename(columns={old_name: param})
                sample = sample[day_cat]
                tmp.append(sample)

            # Per ogni giorno genero un dataframe di 24h contenente tutti i valori
            temp_df = pd.concat(tmp, axis=1)
            samples.append(temp_df)

        return samples, days_serie

    @staticmethod
    def _day_distribution(cat_days: dict[Category, list[str]]):
        """
        Calcola la distribuzione delle categorie dei giorni
        :param cat_days: Dizionario con lista di giorni categorizzati
        :return: Lista delle categorie Lista delle probabilitá delle categorie
        """
        # Calcolo delle frequenze per categoria
        frequenze = {categoria: len(giorni) for categoria, giorni in cat_days.items()}

        # Calcolo del numero totale di giorni
        totale_giorni = sum(frequenze.values())

        # Calcolo delle probabilità per categoria
        probabilita = {categoria: frequenza / totale_giorni for categoria, frequenza in frequenze.items()}

        categorie = list(probabilita.keys())
        pesi_probabilita = np.array(list(probabilita.values()))

        return categorie, pesi_probabilita

    @staticmethod
    def _category_matrix(dati_orari: list[dict[pd.Timestamp, float]],
                         giorni_per_categoria: dict[Category, list[str]]) -> \
            dict[Category, pd.DataFrame]:
        """
        Crea un DataFrame per ogni categoria. Ogni riga rappresenta un giorno univoco all'interno della categoria,
        le colonne sono i timestamp specifici di quel giorno, e i valori sono i dati associati a ciascun timestamp.
        """
        # Convertire i dati orari in DataFrame
        df_orari = pd.DataFrame(dati_orari)
        df_orari['timestamp'] = pd.to_datetime(df_orari['timestamp'], unit='s')
        df_orari['data'] = df_orari['timestamp'].dt.strftime('%Y-%m-%d')

        category = {}

        for categoria, date in giorni_per_categoria.items():
            # Creare una lista per raccogliere i dati per la categoria corrent

            # Creare un elenco vuoto per tenere traccia dei nomi delle colonne (timestamp univoci)
            day_data = {}

            # Iterare su ogni giorno specificato per la categoria
            for data_str in date:
                # Filtrare i dati per la data corrente
                dati_del_giorno = df_orari[df_orari['data'] == data_str]
                dati_del_giorno = dati_del_giorno.sort_values(by='timestamp')

                row = {}
                for index, riga_oraria in dati_del_giorno.iterrows():
                    row[riga_oraria.iloc[0].strftime("%X")] = riga_oraria.iloc[1]

                day_data[data_str] = row

            category[categoria] = day_data

        for cat, data_rows in category.items():
            temp_df = pd.DataFrame.from_dict(data_rows, orient="index").reset_index()
            category[cat] = temp_df

        return category

    @staticmethod
    def _generate_synt_data(matrix: dict[Category, pd.DataFrame]) -> dict[Category, pd.DataFrame]:
        """
        Genera sample per ogni categoria
        :param matrix: Dizionario con i valori dei giorni categorizzati
        :return: Dizionario con 1 sample per ogni categoria
        """
        synt = {}

        for category, matrice in matrix.items():
            syn_data = []

            if matrice.empty:
                continue

            # Calcolo della risoluzione temporale in minuti (-1 per rimuovere l'index
            N_times = matrice.shape[1] - 1
            resolution_minutes = 24 * 60 // N_times

            # Crea l'elenco dei timestamp per l'intero giorno basato sulla risoluzione
            all_timestamps = [f"{hour:02d}:{minute:02d}:00" for hour in range(24) for minute in
                              range(0, 60, resolution_minutes)]

            # Determina l'intervallo di tempo di interesse analizzando i valori delle colonne
            timestamps_di_interesse = []
            for col in all_timestamps:
                # Controlla se ci sono valori significativi nella colonna (escludendo i valori pari a zero)
                if not matrice[col].eq(0).all():  # Se la colonna ha almeno un valore non zero
                    timestamps_di_interesse.append(col)

            # Determina l'indice del primo e dell'ultimo timestamp di interesse
            if timestamps_di_interesse:
                primo_timestamp = timestamps_di_interesse[0]
                ultimo_timestamp = timestamps_di_interesse[-1]
                # Filtra le colonne del DataFrame matrice basandosi sull'intervallo trovato
                indice_inizio = all_timestamps.index(primo_timestamp)
                indice_fine = all_timestamps.index(ultimo_timestamp)
                selected_timestamps = all_timestamps[indice_inizio:indice_fine + 1]
                df_filtered = matrice[selected_timestamps]
            else:
                # Se nessuna colonna ha valori significativi, considera tutto il DataFrame
                selected_timestamps = all_timestamps
                df_filtered = matrice

            if df_filtered.empty:
                continue

            # Calcolare il min e il max per ogni ora (colonna) e popolare la matrice
            for i, ora in enumerate(selected_timestamps):

                if i == 0:
                    val = matrice.loc[:, selected_timestamps[i]]  # Vettore relativo al singolo timestamp
                    val = val.dropna()  # Ignora le righe con Null

                    mean = val.mean()

                    if val.empty:
                        std = 0
                    else:
                        std = val.std()

                    if math.isnan(std):
                        std = 0

                    syn_data.append(abs(np.random.normal(loc=mean, scale=std)))
                else:
                    val_1 = matrice.loc[:, selected_timestamps[i - 1]]  # Vettore relativo al timestamp precedente
                    val_1 = val_1.dropna()  # Ignora i NaN

                    val = matrice.loc[:, selected_timestamps[i]]  # Vettore relativo al singolo timestamp
                    val = val.dropna()  # Ignora i NaN

                    # mean = val_1.mean(axis=1)

                    if val_1.empty:
                        std = 0
                    else:
                        std = val_1.std()

                    if math.isnan(std):  # Avoids NaN
                        std = 0

                    N = len(val_1)

                    upper = syn_data[i - 1] + (1.96 * std / np.sqrt(N))
                    lower = syn_data[i - 1] - (1.96 * std / np.sqrt(N))

                    min_val = val_1.min()
                    max_val = val_1.max()

                    if upper > max_val:
                        upper = max_val
                    if lower < min_val:
                        lower = min_val

                    if upper < lower:  # Caso in cui il lower risulti negativo a causa di valori troppo piccoli
                        upper = max_val

                    temp = pd.DataFrame({
                        selected_timestamps[i - 1]: val_1,
                        selected_timestamps[i]: val
                    })

                    temp = temp.loc[
                        (lower <= temp[selected_timestamps[i - 1]]) & (temp[selected_timestamps[i - 1]] <= upper)]

                    if len(temp[selected_timestamps[i]]) == 0:
                        syn_data.append(syn_data[i - 1])
                    else:
                        std = np.std(temp[ora])
                        if math.isnan(std):
                            std = 0
                        s = np.random.normal(loc=np.mean(temp[ora]), scale=std)

                        if s >= np.max(temp[ora].values):
                            s = np.max(temp[ora].values)

                        if s <= np.min(temp[ora].values):
                            s = np.min(temp[ora].values)

                        syn_data.append(abs(s))

            df_syn_data = pd.DataFrame()
            df_syn_data['syn_data'] = syn_data
            df_syn_data['index'] = selected_timestamps
            df_syn_data = df_syn_data.set_index('index').reset_index()

            df_syn_data['syn_data'] = df_syn_data['syn_data'].abs()

            # Fillin missing timestamps
            extended = pd.DataFrame(0, index=all_timestamps, columns=["syn_data_ext"])
            df_syn_data = df_syn_data.set_index('index')
            extended_fix = pd.concat([df_syn_data, extended], axis=1).sort_index()

            # Cleaning NaN and negative values
            extended_fix = extended_fix.drop('syn_data_ext', axis=1).fillna(0)
            df_syn_data = extended_fix
            synt[category] = df_syn_data
        return synt
