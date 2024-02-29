import pandas as pd
import plotly.graph_objects as go


def metric_compare(file_csv_1: str = "", file_csv_2: str = "", label1: str = "", label2: str = "",
                   metric: str = "RMSE", ):
    # Carica i dati dai file CSV
    df1 = pd.read_csv(file_csv_1, decimal='.')
    df2 = pd.read_csv(file_csv_2, decimal='.')

    filter_col = [df1.columns[0]]
    filter_col += [col for col in df1.columns if col == metric]

    df_merged = pd.merge(df1[filter_col], df2[filter_col], on=df1.columns[0], suffixes=(f' {label1}', f' {label2}'))

    metrics = [col for col in df_merged.columns if col not in df1.columns[0]]

    fig = go.Figure()

    for i, col in enumerate(metrics):
        fig.add_trace(go.Bar(x=df_merged['Parameters'], y=df_merged[f"{col}"], name=f'{col}'))

    fig.update_layout(barmode='group', title_text='Models comparison', xaxis_title="Parameter",
                      yaxis_title="Values")

    # Mostra il grafico
    fig.show(renderer='plotly_mimetype+notebook')
