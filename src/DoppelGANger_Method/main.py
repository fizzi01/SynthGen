import sys

import pandas as pd
from pvlib import pvsystem
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from tqdm import trange
import calendar
from datetime import datetime

import DataLoad as dl
import DataPlot as dp

LOCATION = Location(40.3548, 18, 0, 0, 'Lecce')

loader = dl.Data(data_path="open-meteo-Lecce_new.csv")
categorical_cols = ["Mese"]
numerical_cols = [col for col in loader.data.columns if col not in categorical_cols]
mba_data = loader.prepare_data(numerical_cols, categorical_cols)




model_args = ModelParameters(batch_size=100,
                             lr=0.001,
                             betas=(0.2, 0.9),
                             latent_dim=20,
                             gp_lambda=2,
                             pac=1)

train_args = TrainParameters(epochs=5000,
                             sequence_length=24,
                             sample_length=6,
                             rounds=1,
                             measurement_cols=numerical_cols)

# Training the DoppelGANger synthesizer

#model_dop_gan = TimeSeriesSynthesizer(modelname='doppelganger',model_parameters=model_args)
#model_dop_gan.fit(mba_data, train_args, num_cols=numerical_cols, cat_cols=categorical_cols)
model_dop_gan = TimeSeriesSynthesizer.load('DoppelGan_date_test3.pkl')


# Generating new synthetic samples
#model_dop_gan.save('DoppelGan_date_test3.pkl')
synth_data = model_dop_gan.sample(n_samples=int(365))
synth_df = pd.concat(synth_data, axis=0)
#Cleaning negative values
synth_df[numerical_cols] = synth_df[numerical_cols].clip(lower=0)
synth_df_tmp = loader.processor.reverse_transform(synth_df[numerical_cols].values)
synth_df_cat = pd.DataFrame(synth_df[categorical_cols],columns=categorical_cols).reset_index(drop=True)
synth_df_num = pd.DataFrame(synth_df_tmp, columns=numerical_cols).reset_index(drop=True)


synth_df = pd.concat([synth_df_num, synth_df_cat], axis=1)
synth_df["Mese"] = synth_df["Mese"].apply(lambda x: datetime.strptime(x, '%B').month)
synth_df = synth_df.rename_axis("index").sort_values(by = ["Mese", "index"])
synth_df["Mese"] = synth_df["Mese"].apply(lambda x: calendar.month_name[x])
synth_df = synth_df.reset_index(drop=True)

#Converting months
#synth_df["Mese"] = synth_df["Mese"].apply(lambda x: calendar.month_abbr[round(x)])


dp.plty_plot_dataframes([synth_df],title="Synth DoppelGANger")



#PVLIB
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


# Convertire gli orari in timestamp completi
synth_df.index = loader.data.head(synth_df.shape[0]).index
df = synth_df
# Configurazione del sistema PV e della location

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
results["p_AC"] = modelchain.results.ac

complete_sample.append(results)

complete_sample = pd.concat(complete_sample)
dp.plty_plot_dataframes([complete_sample], title="Syntethic Panel Data")
