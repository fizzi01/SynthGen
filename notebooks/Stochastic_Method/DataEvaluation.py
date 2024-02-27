"""
Different methods for data evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from Utils import Category

METRICS = ["RMSE", "MSE"]


def get_metrics(samples: list[pd.DataFrame], day_distr: np.array(Category), param_matrix: dict, params: list):
    """
    Per ogni sample, fa una valutazione (media) sui giorni che appartengono alla stessa categoria, così per ogni sample
    e per ogni parametro dei samples. Valutazione finale è la media relativa ciascun parametro.
    :param samples: Synthetic data list
    :param day_distr: Category list for every sample
    :param param_matrix: Generator matrix with days per category
    :param params: List of parameters
    :return: Dataframe with metrics for each parameter
    """
    metrics_val = {}

    for metric in METRICS:
        metrics_val[metric] = {}
        for parameter in params:
            metrics_val[metric][parameter] = []

    for metric in METRICS:
        for sample, cat in zip(samples, day_distr):
            for parameter in param_matrix.keys():

                tmp_metrics = []

                # Recupero del parametro tutti i valori della categoria
                # Ottengo un dataframe con 24 righe e tante colonne quanti i giorni (trasposta)
                real = param_matrix[parameter][cat].T.iloc[1:].reset_index(drop=True)
                for day in real.columns:
                    serie = real[day]
                    serie = pd.to_numeric(serie)
                    # Fare evaluation per ogni giorno della categoria e poi farne la media
                    tmp_metrics.append(evaluate(serie, sample[parameter], metric))

                # Conoscendo la categoria del sample, faccio l'evaluate su tutti i giorni appartenenti alla stessa categoria e ne faccio la media
                tmp_mean = metrics_val[metric][parameter]
                tmp_mean.extend(tmp_metrics)
                metrics_val[metric][parameter] = tmp_mean

    for metric in METRICS:
        for parameter in param_matrix.keys():
            tmp_metrics = metrics_val[metric][parameter]
            tmp_mean = np.mean(np.array(tmp_metrics))
            metrics_val[metric][parameter] = tmp_mean

    return pd.DataFrame.from_dict(metrics_val, orient='columns')


def evaluate(real, synth, metric):
    if metric == "SMAPE":
        return smape(real, synth)
    elif metric == "MAPE":
        return mape(real, synth)
    elif metric == "RMSE":
        return rmse(real, synth)
    elif metric == "MSE":
        return mse(real, synth)


def prepare_data(real, synth):
    # Check if is an array, if not fix it
    if not all([isinstance(real, np.ndarray),
                isinstance(synth, np.ndarray)]):
        real, synth = np.array(real), np.array(synth)

    return real, synth


def rmse(real, synth):
    real, synth = prepare_data(real, synth)
    rmse_val = mean_squared_error(y_true=real, y_pred=synth, squared=False)

    return rmse_val


def mse(real, synth):
    real, synth = prepare_data(real, synth)
    mse_val = mean_squared_error(y_true=real, y_pred=synth, squared=True)

    return mse_val


def mape(real, synth):
    real, synth = prepare_data(real, synth)
    mape_val = round(np.mean(np.abs(synth - real) / (np.abs(synth))) * 100, 3)
    return mape_val


def smape(real, synth):
    real, synth = prepare_data(real, synth)
    smape_val = round(np.mean(np.abs(synth - real) / ((np.abs(synth)) + np.abs(real)) / 2) * 100, 3)
    return smape_val
