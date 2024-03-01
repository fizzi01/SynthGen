import sys
from os import path

from datetime import datetime
import calendar
import pandas as pd

from pvlib import pvsystem

from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.synthesizers import ModelParameters

from tqdm import trange

from DataPlot import *

import timegan_method.DataProcessor as dp
import timegan_method.PvModel as mdl
import timegan_method.DataEvaluation as de
from timegan_method.DataLoad import *

import matplotlib.pyplot as plt
import numpy as np


LOCATION = Location(40.3548, 18, 0, 0, 'Lecce')

SEQ_LENGHT = 24  # Dimensione della sliding window 24
PATH = "open-meteo-Lecce_new.csv"
n_seq = 8  # Numero di variabili
hidden_dim = 24

gamma = 1
batch_size = 128
learning_rate = 0.005
noise_dim = 32
dim = 128

EPOCHS = 10000
SAMPLES_N = 356

data_df = Data(data_path=PATH, seq_len=SEQ_LENGHT)
cat_col = ["Mese"]
SAMPLES_N = len(data_df.data)

#Aggiorno numero di variabili per considerare il onehot encoding per le variabili categoriche


# Preparo i parametri per il generatore discriminatore & co
gan_args = ModelParameters(batch_size=batch_size, lr=learning_rate, noise_dim=noise_dim, layers_dim=dim)

# Controllo se già presente modello trained
#retrain = input('Retrain? (y/n) ').lower().replace(" ", "")
#model = input('Model path:')
#metadata = input('Model metadata:')
model = "final_model.pkl"
metadata = "final_meta.pkl"

"""if retrain == 'n':"""
if path.exists(model) and path.exists(metadata):
    synth = TimeSeriesSynthesizer.load(model)
    processor = dp.DataProcessor().load_metadata(metadata)
    print()
    print(f"Model trained - {model}\nmetadata - {metadata}")
else:
    raise FileNotFoundError

"""else:
    data = data_df.prepare_data([c for c in data_df.data.columns if c not in cat_col], cat_col)
    data_df.save(metadata)
    processor = data_df.processor
    n_seq = data[0].shape[1]

    synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=SEQ_LENGHT, n_seq=n_seq, gamma=gamma)

    synth.train(data, train_steps=EPOCHS)
    synth.save(model)"""


print()
generate = input('Generate a sequence? (y/n) ').lower().replace(" ", "")

if generate == 'n':
    file_toload = input("Saved results file:")
    if path.exists(file_toload):
        synth_results = np.load(file_toload)
    else:
        raise FileNotFoundError("Specific results file does not exist")
else:
    synth_results = synth.sample(SAMPLES_N)
    # Saving samples
    np.save(f"synth_results_{datetime.now().minute}.npy", synth_results)

data = data_df.prepare_data(processor.num_columns, processor.cat_columns)


eval_res = de.DataEvaluation(data, synth_results, SEQ_LENGHT, 500)
eval_res.evaluate_data()

# Get the complete timeseries, by inverting preprocessing so denormalize it
synth_results = synth_results.reshape(-1, processor.col_number)
synth_results = processor.reverse_transform(synth_results)
synth_df = pd.DataFrame(synth_results, columns=processor.get_columns())

# Fixing dtypes
synth_df[processor.num_columns] = synth_df[processor.num_columns].apply(pd.to_numeric)

# SAMPLES WINDOWS TO PLOT
samples_to_plot = []
# Numero totale di righe nel dataframe
total_rows = len(synth_df)

m_resolution = int(24 * 60 / int((data_df.data.index[1] - data_df.data.index[0]).seconds / 60))

# Iterare attraverso il DataFrame a blocchi di 24 righe
for start in range(0, total_rows, m_resolution):
    # Verifica che ci siano abbastanza righe per una finestra completa di 24 ore
    if start + m_resolution > total_rows:
        break  # Esce dal ciclo se non ci sono abbastanza righe per una nuova finestra

    # Estrai il blocco di 24 righe
    finestra = synth_df.iloc[start:start + m_resolution]
    finestra = finestra.reset_index(drop=True)

    samples_to_plot.append(finestra)


# EXTRACTING DAY FRAMES
for plots in range(5):
    index = np.random.randint(len(samples_to_plot))
    plty_plot_dataframes([samples_to_plot[index]], title=f"Sample number {index}")

# Lista per salvare le finestre di 24 ore che corrispondono al pattern
finestre_valide = dp.extract_samples(real=data_df.data,synth=synth_df)

synth_tot = pd.concat(finestre_valide).reset_index(drop=True)
synth_tot["Mese"] = synth_tot["Mese"].apply(lambda x: datetime.strptime(x, '%B').month)
synth_tot = synth_tot.rename_axis("index").sort_values(by=["Mese", "index"])
synth_tot["Mese"] = synth_tot["Mese"].apply(lambda x: calendar.month_name[x])
# Resetting index order
synth_tot = synth_tot.reset_index(drop=True)

pattern_to_plot = []
# Iterare attraverso il DataFrame a blocchi di 24 righe
for start in range(0, 2, m_resolution):
    # Verifica che ci siano abbastanza righe per una finestra completa di 24 ore
    if start + m_resolution > total_rows:
        break  # Esce dal ciclo se non ci sono abbastanza righe per una nuova finestra

    # Estrai il blocco di 24 righe
    finestra = synth_tot.iloc[start:start + m_resolution]
    finestra = finestra.reset_index(drop=True)

    plty_plot_dataframes([finestra], title=f"Sample number {start}")

#plty_plot_dataframes([synth_tot], title="Synthetic Data")


print(f"Samples found: {len(finestre_valide)}")



# Recupero i mesi che ricorrono nei samples estratti
month_list = synth_tot["Mese"].unique()
month_samples = []
# Per ogni i esimo mese, recupero i samples
for month in month_list:
    tmp = synth_tot.loc[synth_tot["Mese"] == month]
    month_samples.append(tmp)

panel = mdl.PVModel(location=LOCATION)
complete_sample = panel.run_model(sample=finestre_valide)

plty_plot_dataframes([complete_sample], title="Syntethic Panel Data")

print()