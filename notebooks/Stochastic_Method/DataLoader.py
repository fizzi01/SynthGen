# Contiene funzioni per il loading dei dati dai file CSV
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataLoader:

    def __init__(self, filename: str):
        self.filename = filename
        self.data = self._read_csv()

    def _read_csv(self) -> pd.DataFrame:
        """
        Legge il file CSV e lo memorizza in un DataFrame per successive elaborazioni
        """
        df = pd.read_csv(self.filename)

        # Converting in datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp').sort_index()

        return df

    def list_parameters(self) -> list[str]:
        """
        Restituisce tutti i parametri possibili del file CSV
        :return: Lista dei parametri
        """
        return self.data.columns.tolist()

    def extract_parameter(self, parameter: str, year: int | list = None, month: int = None, resolution: int = None):
        """
        Estrae i dati per un parametro specificato, insieme ai timestamp corrispondenti.
        Può filtrare i dati per un mese specifico se fornito.

        :param resolution: Risoluzione per upscaling del dataset (None = dataset current resolution)
        :param parameter: Il nome del parametro da estrarre.
        :param year: L'anno per il quale filtrare i dati (opzionale).
        :param month: Il mese per il quale filtrare i dati (opzionale).
        :return: Una lista di dizionari contenenti il timestamp e il valore del parametro.
        """
        if parameter not in self.data.columns:
            print(f"Warning: Il parametro '{parameter}' non è stato trovato.")
            return []

        if resolution == 60:
            resolution = None


        #self.data.reset_index(drop=False)
        #self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='s')
        original_data = self.data.copy(deep=True)
        self.data = self.data.reset_index(drop=False)

        # Filtrare per anno e mese se specificato
        if year is not None and month is not None:
            filtered_dfs = []  # Lista per raccogliere i DataFrame filtrati

            # Se 'year' è una lista, filtra per ciascun anno nella lista
            if isinstance(year, list):
                for y in year:
                    start_date = f"{y}-{month:02d}-01"
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd()
                    filtered_df = self.data[
                        (self.data['timestamp'] >= start_date) & (self.data['timestamp'] <= end_date)]

                    # Single cleaning and sampling
                    filtered_df = filtered_df[['timestamp', parameter]].dropna()
                    if resolution is not None:
                        resample = filtered_df.set_index('timestamp').resample(f'{resolution}min')
                        resample = resample.interpolate(method='spline', order=2)

                        resample = resample.reset_index()

                    else:
                        resample = filtered_df

                    filtered_dfs.append(resample)

                # Combina i DataFrame filtrati
                filtered_df = pd.concat(filtered_dfs)
            else:
                # Filtraggio per un singolo anno se 'year' non è una lista
                start_date = f"{year}-{month:02d}-01"
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd()
                filtered_df = self.data[(self.data['timestamp'] >= start_date) & (self.data['timestamp'] <= end_date)]

                extracted_data = filtered_df[['timestamp', parameter]].dropna()
                if resolution is not None:
                    resample = extracted_data.set_index('timestamp').resample(f'{resolution}min')
                    resample = resample.interpolate(method='spline', order=3)

                    resample = resample.reset_index()

                else:
                    resample = extracted_data

                filtered_df = resample
        else:
            # Se anno o mese non sono specificati, ritorna il DataFrame originale
            filtered_df = self.data
            # Selezionare solo le righe con il parametro specificato e convertirle in lista di dizionari .to_dict('records')
            extracted_data = filtered_df[['timestamp', parameter]].dropna()
            if resolution is not None:
                resample = extracted_data.set_index('timestamp').resample(f'{resolution}min')
                resample = resample.interpolate(method='spline', order=3)

                resample = resample.reset_index()

            else:
                resample = extracted_data
            filtered_df = resample

        resample = filtered_df.to_dict('records')

        for val in resample:
            val[parameter] = abs(float(val[parameter]))

        # extracted_data = extracted_data.to_dict('records')
        extracted_data = resample

        # Reconvertire timestamp in Unix per coerenza con la specifica iniziale
        for item in extracted_data:
            item['timestamp'] = int(item['timestamp'].timestamp())
            item[parameter] = float(item[parameter])  # Assicurarsi che i valori siano float

        # Restoring original data
        self.data = original_data.copy(deep=True)

        return extracted_data


def plot_extracted_data(data_list: list[dict], year: int = 0, month: int = 0, day: int = 0):
    """
    Genera un plot dei dati estratti, mostrando solo le linee per un singolo giorno specificato
    attraverso anno, mese e giorno.

    :param data: Lista di dizionari contenenti il timestamp e il valore del parametro.
    :param year: Anno da considerare per il filtraggio.
    :param month: Mese da considerare per il filtraggio.
    :param day: Giorno del mese da considerare per il filtraggio.
    """
    plt.figure(figsize=(24, 6))
    colors = plt.cm.Paired(np.linspace(0, 1, len(data_list)))
    title = "Plot extracted data"
    if not data_list:
        print("Nessun dato disponibile per il plot.")
        return
    for i, data in enumerate(data_list):
        df = pd.DataFrame(data)
        parameter = df.columns[1]  # Assumendo che la prima colonna sia 'timestamp'
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df[parameter] = pd.to_numeric(df[parameter])
        df.sort_values('timestamp', inplace=True)

        # Costruire la data specifica da filtrare
        if year != 0 and month != 0 or day != 0:
            specific_date = pd.Timestamp(year=year, month=month, day=day)
            next_day = pd.Timestamp(specific_date) + pd.offsets.MonthEnd(n=1)

            # Filtrare i dati per il giorno specificato
            df = df[(df['timestamp'] >= specific_date) & (df['timestamp'] <= next_day)]

            title = f"Plot {specific_date} , {next_day}"

        if df.empty:
            print(f"Nessun dato disponibile per {year}-{month:02d}-{day:02d}.")
            return

        plt.plot(df['timestamp'], df[parameter], label=parameter, marker='*', linestyle='-', color=colors[i])

    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel(f"Value", fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.legend()

    if year != 0 and month != 0 and day != 0:
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %X'))
    else:
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Migliora la formattazione delle date sull'asse x

    plt.show()
