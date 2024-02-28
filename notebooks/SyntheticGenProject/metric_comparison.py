import pandas as pd
import plotly.graph_objects as go


def confronta_csv_plotly(file_csv_1, file_csv_2, file1_name, file2_name):
    # Carica i dati dai file CSV
    df1 = pd.read_csv(file_csv_1, decimal='.')
    df2 = pd.read_csv(file_csv_2, decimal='.')

    df_merged = pd.merge(df1, df2, on=df1.columns[0], suffixes=(f' {file1_name}', f' {file2_name}'))

    metrics = [col for col in df_merged.columns if col not in df1.columns[0]]

    fig = go.Figure()

    for i, col in enumerate(metrics):
        fig.add_trace(go.Bar(x=df_merged['Parameters'], y=df_merged[f"{col}"], name=f'{col}'))

    fig.update_layout(barmode='group', title_text='Models comparison', xaxis_title="Parameter",
                      yaxis_title="Values")

    # Mostra il grafico
    fig.show(renderer='plotly_mimetype+notebook')
