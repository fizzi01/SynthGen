import pandas as pd
import plotly.graph_objects as go


def confronta_csv_plotly(file_csv_1, file_csv_2, file1_name, file2_name):
    # Carica i dati dai file CSV
    df1 = pd.read_csv(file_csv_1, decimal='.')
    df2 = pd.read_csv(file_csv_2, decimal='.')

    # Unisci i DataFrame su "Parameter" per confrontare i valori
    df_merged = pd.merge(df1, df2, on="Parameter", suffixes=('_1', '_2'))

    # Creazione del grafico
    fig = go.Figure()

    # Aggiungi barre per RMSE e MSE del primo CSV
    fig.add_trace(go.Bar(x=df_merged['Parameter'], y=df_merged['RMSE_1'], name=f'RMSE {file1_name}', marker_color='blue'))
    fig.add_trace(go.Bar(x=df_merged['Parameter'], y=df_merged['MSE_1'], name=f'MSE {file1_name}', marker_color='lightblue'))

    # Aggiungi barre per RMSE e MSE del secondo CSV
    fig.add_trace(go.Bar(x=df_merged['Parameter'], y=df_merged['RMSE_2'], name=f'RMSE {file2_name}', marker_color='red'))
    fig.add_trace(go.Bar(x=df_merged['Parameter'], y=df_merged['MSE_2'], name=f'MSE {file2_name}', marker_color='pink'))

    # Personalizza il layout
    fig.update_layout(barmode='group', title_text='Confronto Parametri tra File CSV', xaxis_title="Parameter",
                      yaxis_title="Valori")

    # Mostra il grafico
    fig.show()

file_csv_1 = 'Metrics/Cartel1.csv'
file_csv_2 = 'Metrics/stoch_metrics.csv'

# Chiama il metodo
confronta_csv_plotly(file_csv_1, file_csv_2,"DoppelGANger (Lecce)","Stochastic")