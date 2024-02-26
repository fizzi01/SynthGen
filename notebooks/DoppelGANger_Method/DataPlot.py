import plotly.graph_objects as go
import pandas as pd


def plty_plot_dataframes(dataframes, title: str = "Plot"):
    """
    Fa il plot di una lista di DataFrame su un unico grafico.

    :param title: Titolo dataframe
    :param dataframes: Lista di DataFrame da plottare.
    """

    fig = go.Figure()

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Values',
        legend_title='Prameters',
        hovermode='closest'
    )

    for i,df in enumerate(dataframes):
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f"{i}_{col}", opacity=float(1/(i+1))))

    # Abilita la possibilità di mostrare/nascondere ogni linea dalla legenda
    fig.update_layout(legend=dict(itemsizing='constant'))
    # Mostra il grafico
    fig.show()
    #fig.show(renderer='plotly_mimetype+notebook')
