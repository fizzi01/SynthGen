import sys
from datetime import datetime

import pandas as pd
from pvlib import pvsystem
from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.timeseries import TimeGAN

from os import path
from DataLoad import *
from DataPlot import *

import matplotlib.pyplot as plt
import numpy as np
import DataEvaluation as de
import tqdm
from tqdm import trange

from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

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

data_df = Data(data_path=PATH, seq_len=SEQ_LENGHT)
cat_col = ["Mese"]
data = data_df.prepare_data([c for c in data_df.data.columns if c not in cat_col], cat_col)

print(len(data), data[0].shape)
#Aggiorno numero di variabili per considerare il onehot encoding per le variabili categoriche
n_seq = data[0].shape[1]

# Preparo i parametri per il generatore discriminatore & co
gan_args = ModelParameters(batch_size=batch_size, lr=learning_rate, noise_dim=noise_dim, layers_dim=dim)

# Controllo se già presente modello trained
retrain = input('Retrain? (y/n) ').lower().replace(" ", "")

if path.exists("synth_gen.pkl") and not retrain == 'y':
    synth = TimeGAN.load("synth_gen.pkl")
    print()
    print(f"Model trained: epochs={EPOCHS}, variables={n_seq}")

else:
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=SEQ_LENGHT, n_seq=n_seq, gamma=gamma)

    synth.train(data, train_steps=EPOCHS)
    synth.save("synth_gen_.pkl")

print()
generate = input('Generate a sequence? (y/n) ').lower().replace(" ", "")

if generate == 'n' and path.exists("synth_results.npy"):
    synth_results = np.load("synth_results.npy")
else:
    synth_results = synth.sample(len(data))
    # Saving samples
    np.save(f"synth_results_{datetime.now().minute}.npy", synth_results)

# Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 10))
axes = axes.flatten()

time = list(range(1, 25))
obs = np.random.randint(len(data))

for j, col in enumerate(data_df.data.columns):
    df = pd.DataFrame({'Real': data[obs][:, j],
                       'Synthetic': synth_results[obs][:, j]})
    df.plot(ax=axes[j],
            title=col,
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()
plt.show()

eval_res = de.DataEvaluation(data, synth_results, SEQ_LENGHT, 500)
eval_res.evaluate_data()

# Get the complete timeseries, by inverting preprocessing so denormalize it
synth_results = synth_results.reshape(-1, 8)
synth_results = data_df.processor.reverse_transform(synth_results)
synth_df = pd.DataFrame(synth_results, columns=data_df.data.columns)

# EXTRACTING DAY FRAMES ....................

# Ho bisogno di alcune metriche per l'individuazione del pattern che corrisponde a una finestra di 24 ore.
# Selezione basata sull'ora per il calcolo delle statistiche
# Ore diurne: dalle 6 alle 18
ghi_diurni = data_df.data['shortwave_radiation_instant (W/m²)'].between_time('05:00', '19:00')

# Calcolo delle soglie
soglia_bassa = ghi_diurni.quantile(0.25)

# Lista per salvare le finestre di 24 ore che corrispondono al pattern
finestre_valide = []
# Numero totale di righe nel dataframe
total_rows = len(synth_df)

resolution = int(60 / (data_df.data.index[1] - data_df.data.index[0]).seconds * 60)


def verifica_pattern(finestra):
    # Assumiamo che 'finestra' sia una porzione del DataFrame con indici datetime
    ore_diurne = finestra[5 * resolution:18 * resolution]  # ('06:00', '18:00')
    ore_notturne_pre = finestra[:5 * resolution]  # 00 - 5:00
    ore_notturne_post = finestra[18 * resolution:]  # ('18:00', '23:00')

    # Controlla il pattern usando le soglie calcolate precedentemente
    if ore_notturne_pre.max() <= soglia_bassa and ore_notturne_post.max() <= soglia_bassa and not ((finestra < soglia_bassa).all()):
        return True
    return False


# Iterare attraverso il DataFrame a blocchi di 24 righe
for start in tqdm.tqdm(range(0, total_rows, 24), file=sys.stdout, desc="Extracting samples"):
    # Verifica che ci siano abbastanza righe per una finestra completa di 24 ore
    if start + 24 > total_rows:
        break  # Esce dal ciclo se non ci sono abbastanza righe per una nuova finestra

    # Estrai il blocco di 24 righe
    finestra = synth_df.iloc[start:start + 24]
    finestra = finestra.reset_index(drop=True)

    # Supponiamo che la logica di verifica del pattern venga definita qui
    # Questo è un placeholder per il tuo criterio specifico di verifica del pattern
    # Potresti aver bisogno di personalizzare questa parte in base ai tuoi dati
    if verifica_pattern(finestra['shortwave_radiation_instant (W/m²)']):
        finestre_valide.append(finestra)


print(f"Samples found: {len(finestre_valide)}")
# TODO Bisognerebbe cercare di valutare a quale mese corrispondono ciascun sample,
#  in modo che la simulazione sia più accurata


# Generating random day sequence (needed by pvlib)
date_complete = pd.date_range(
    start=f"{np.random.randint(low=2022, high=2023)}-{np.random.randint(low=1, high=12):02d}-01",
    periods=len(finestre_valide))
date_complete += pd.DateOffset(days=np.random.randint(0, 31))
LOCATION = Location(40.3548, 18, 0, 0, 'Lecce')

# Specifica per inverter e modulo
modules = pvsystem.retrieve_sam('SandiaMod')
module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']

inverters = pvsystem.retrieve_sam('cecinverter')
inverter_parameters = inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

sistema_pv = PVSystem(surface_tilt=20, surface_azimuth=180,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter_parameters,
                      temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'])
modelchain = ModelChain(sistema_pv, LOCATION, aoi_model='physical', spectral_model='no_loss')

complete_sample = []


for i in trange(len(finestre_valide), file=sys.stdout, desc="Generating energy data"):
    df = finestre_valide[i]
    date = date_complete[i]
    # Convertire gli orari in timestamp completi
    df.index = pd.date_range(start=date, periods=24 / resolution, freq=f'{resolution * 60}min')
    # Configurazione del sistema PV e della location

    # Preparazione dei dati atmosferici
    weather = pd.DataFrame({
        'ghi': df['shortwave_radiation_instant (W/m²)'],  # Global Horizontal Irradiance
        'dhi': df['diffuse_radiation_instant (W/m²)'],  # Diffuse Horizontal Irradiance
        'dni': df['direct_normal_irradiance_instant (W/m²)'],  # Direct Normal Irradiance
        'temp_air': df['temperature_2m (°C)'],
        # 'wind_speed': df['wind_speed_10m (m/s)']
    }, index=df.index)

    # Calcolo della produzione energetica
    modelchain.run_model(weather)
    results = pd.DataFrame(modelchain.results.dc)
    results["p_AC"] = modelchain.results.ac

    """poa_energy = modelchain.results.total_irrad['poa_global'].sum() * (1 / 24) / 1000  # Daily POA irradiation in kWh
    dc_energy = modelchain.results.dc['p_mp'].sum() * (1 / 60) / 1000  # Daily DC energy in kWh
    ac_energy = modelchain.results.ac.sum() * (1 / 60) / 1000  # Daily AC energy in kWh

    print('*' * 15, ' Daily Production ', '*' * 15, '\n', '-' * 48)
    print('\tPOA irradiation: ', "%.3f" % poa_energy, 'kWh')
    print('\tDC generation:', "%.3f" % dc_energy, 'kWh')
    print('\tAC generation:', "%.3f" % ac_energy, 'kWh')
    print('-' * 50)"""

    complete_sample.append(results)

complete_sample = pd.concat(complete_sample)
plty_plot_dataframes([complete_sample], title="Syntethic Panel Data")

print()