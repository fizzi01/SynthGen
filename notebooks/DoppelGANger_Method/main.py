import sys
from os import path

import pandas as pd

from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

from pvlib import pvsystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from tqdm import trange
import calendar
from datetime import datetime

import DataLoad as dl
import DataProcessor as dprc
import DataPlot as dp
from DataEvaluation import *

LOCATION = Location(40.3548, 18, 0, 0, 'Lecce')

"""NEW BEST 80024mdl
BATCH = 128
LR = 0.001
LATENT_DIM = 24
EPOCHS = 800
SEQUENCE_LENGTH = 24
SAMPLE_LENGHT = 8
"""


BATCH = 128
LR = 0.001
LATENT_DIM = 24
EPOCHS = 850
SEQUENCE_LENGTH = 24
SAMPLE_LENGHT = 8

loader = dl.Data(data_path="open-meteo-Lecce_new.csv")

categorical_cols = ["Mese"]
numerical_cols = [col for col in loader.data.columns if col not in categorical_cols]


model_args = ModelParameters(batch_size=BATCH,
                             lr=LR,
                             betas=(0.2, 0.9),
                             latent_dim=LATENT_DIM,
                             gp_lambda=2,
                             pac=1)

train_args = TrainParameters(epochs=EPOCHS,
                             sequence_length=SEQUENCE_LENGTH,
                             sample_length=SAMPLE_LENGHT,
                             rounds=1,
                             measurement_cols=numerical_cols)

# Training the DoppelGANger synthesizer
retrain = input('Retrain? (y/n) ').lower().replace(" ", "")
model_path = input('Model path? ')
processor_path = input('Processor path? ')

if retrain == 'n':
    if path.exists(model_path) and path.exists(processor_path):
        # Loading model
        model_dop_gan = TimeSeriesSynthesizer.load(model_path)

        # Loading processor
        processor = dprc.DataProcessor().load_metadata(processor_path)
    else:
        raise FileExistsError("Files not found")

else:
    # Preparing data for training
    mba_data = loader.prepare_data(num_col=numerical_cols, cat_col=categorical_cols)
    processor = loader.processor

    # Saving real data metadata
    loader.save(processor_path)

    model_dop_gan = TimeSeriesSynthesizer(modelname='doppelganger',model_parameters=model_args)
    model_dop_gan.fit(mba_data, train_args, num_cols=numerical_cols, cat_cols=categorical_cols)
    model_dop_gan.save(f"{model_path}")

for _ in trange(0,1, desc="Generating samples"):
    synth_data = model_dop_gan.sample(n_samples=int(365))
synth_df = pd.concat(synth_data, axis=0)

synth_df = processor.reverse_transform(synth_df)

metrics_val = get_metrics(real_df=loader.data, synth_df=synth_df, params=processor.numerical_col)
metrics_val.to_csv("metric.csv",index=False)

dp.plty_plot_dataframes([synth_df],title="Synth DoppelGANger")

# module parameters for the Offgridtec Mono 20W 3-01-001560:
parameters = {
    'Name': 'Offgridtec_001560',
    'BIPV': 'ESG_glass',
    'Date': '04/05/2023',
    'celltype': 'monocrystalline',  # technology
    'Bifacial': True,  # True/False
    'T_NOCT': 45,
    'STC1': 1000,  # Standard Test Conditions power (w/m2)
    'STC2': "AM1.5 Spectrum",  # Standard Test Conditions power
    'STC3': 25,  # Standard Test Conditions power c (Temprature)
    # 'PTC': XXX, # PVUSA Test Condition power
    # 'A_c': XXX, #module cells area in m²
    'N_s': 36,  # number of cells in series
    'Array': "2 * 18",
    'Pmax': 20,  # Wp
    'I_sc_ref': 1.21,  # Short circuit current (ISC)
    'V_oc_ref': 22.3,  # Open circuit voltage (VOC)
    'I_mp_ref': 1.12,  # max. current (IMP)
    'V_mp_ref': 17.8,
    'MSV': 600,  # Maximum system voltage
    'alpha_sc': 0.0045,
    # short circuit current temperature coefficient in A/Δ°C          *** PV producers do not provice this data
    # 'beta_oc': -XXX, #open circuit voltage temperature coefficient in V/Δ°C
    'a_ref': 1.3825,
    # diode ideality factor                                                *** PV producers do not provice this data
    'I_L_ref': 1.21,
    # light or photogenerated current at reference condition in A         *** PV producers do not provice this data
    'I_o_ref': 883e-10,
    # diode saturation current at reference condition in A            *** PV producers do not provice this data
    'R_s': 0.9987,
    # series resistance in Ω                                                 *** PV producers do not provice this data
    'R_sh_ref': 3941.111,
    # shunt resistance at reference condition in Ω                      *** PV producers do not provice this data
    # 'Adjust': XXX, #adjustment to short circuit temperature coefficient in %
    # 'gamma_r': -XXX, #power temperature coefficient at reference condition in %/Δ°
    'Tolerance1': "-3% to 3%",  # Power tolerance
    'Tolerance2': "-5% to 5%",  # Elect. parameters tolerance
    'NOCT': 45,  # +/- 2 degree C
    'temp. coefficient Voc': -0.45,  # %/℃
    'temp. coefficient Isc': -0.45,  # %/℃
    'temp. coefficient P': -0.45,  # %/℃

}
inverters = pvsystem.retrieve_sam('cecinverter')
inverter_parameters = inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

sistema_pv = PVSystem(surface_tilt=20, surface_azimuth=180,
                      module_parameters=parameters,
                      inverter_parameters=inverter_parameters,
                      temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'])
modelchain = ModelChain(sistema_pv, LOCATION, aoi_model='physical', spectral_model='no_loss')

complete_sample = []

# Convertire gli orari in timestamp completi
synth_df.index = loader.data.head(synth_df.shape[0]).index
df = synth_df

# Preparazione dei dati atmosferici
weather = pd.DataFrame({
    'ghi': df['shortwave_radiation_instant (W/m²)'],  # Global Horizontal Irradiance
    'dhi': df['diffuse_radiation_instant (W/m²)'],  # Diffuse Horizontal Irradiance
    'dni': df['direct_normal_irradiance_instant (W/m²)'],  # Direct Normal Irradiance
    'temp_air': df['temperature_2m (°C)'],
    'wind_speed': df['wind_speed_10m (m/s)']
}, index=df.index)

# Calcolo della produzione energetica
modelchain.run_model(weather)
results = pd.DataFrame(modelchain.results.dc)
#results["p_AC"] = modelchain.results.ac

complete_sample.append(results)

complete_sample = pd.concat(complete_sample)
dp.plty_plot_dataframes([complete_sample], title="Syntethic Panel Data")

complete_sample = []

df = loader.data.head(synth_df.shape[0])
# Preparazione dei dati atmosferici
weather = pd.DataFrame({
    'ghi': df['shortwave_radiation_instant (W/m²)'],  # Global Horizontal Irradiance
    'dhi': df['diffuse_radiation_instant (W/m²)'],  # Diffuse Horizontal Irradiance
    'dni': df['direct_normal_irradiance_instant (W/m²)'],  # Direct Normal Irradiance
    'temp_air': df['temperature_2m (°C)'],
    'wind_speed': df['wind_speed_10m (m/s)']
}, index=df.index)

# Calcolo della produzione energetica
modelchain.run_model(weather)
results = pd.DataFrame(modelchain.results.dc)
#results["p_AC"] = modelchain.results.ac

complete_sample.append(results)

complete_sample = pd.concat(complete_sample)
dp.plty_plot_dataframes([complete_sample], title="Real Panel Data")