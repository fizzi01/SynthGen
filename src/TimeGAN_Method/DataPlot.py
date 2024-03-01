import plotly.graph_objects as go
import pandas as pd

def plty_plot_dataframes(dataframes, title: str = "Plot"):
    """
    Fa il plot di una lista di DataFrame su un unico grafico con colori diversi.

    :param title: Titolo dataframe
    :param dataframes: Lista di DataFrame da plottare.
    """

    # Trasforma i DataFrame in un formato long per il plotting
    for df in dataframes:
        # Crea una figura Plotly
        fig = go.Figure()
        # "Sciogliere" il DataFrame da un formato wide a un formato long
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
        # Aggiungi opzioni di layout per zoom e legenda interattiva
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Values',
            legend_title='Parameters',
            hovermode='closest'
        )

        # Abilita la possibilità di mostrare/nascondere ogni linea dalla legenda
        fig.update_layout(legend=dict(itemsizing='constant'))

        # Mostra il grafico
        fig.show()
        #fig.show(renderer='plotly_mimetype+notebook')