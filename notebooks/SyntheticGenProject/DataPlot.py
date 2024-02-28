import plotly.graph_objects as go


def plty_plot_dataframes(dataframes, title: str = "Plot", compare: bool = False):
    """
    Fa il plot di una lista di DataFrame su un unico grafico con colori diversi.

    :param compare: Enable comparison between dataframes
    :param title: Titolo dataframe
    :param dataframes: Lista di DataFrame da plottare.
    """

    if compare:

        fig = go.Figure()

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Values',
            legend_title='Prameters',
            hovermode='closest'
        )

        for i, df in enumerate(dataframes):
            for col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], mode='lines', name=f"{i}_{col}", opacity=float(1 / (i + 1))))

        # Abilita la possibilità di mostrare/nascondere ogni linea dalla legenda
        fig.update_layout(legend=dict(itemsizing='constant'))

        # Mostra il grafico
        fig.show(renderer='plotly_mimetype+notebook')
    else:
        for df in dataframes:
            fig = go.Figure()

            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))

            fig.update_layout(
                title=title,
                xaxis_title='Time',
                yaxis_title='Values',
                legend_title='Prameters',
                hovermode='closest'
            )

            # Abilita la possibilità di mostrare/nascondere ogni linea dalla legenda
            fig.update_layout(legend=dict(itemsizing='constant'))

            # Mostra il grafico
            fig.show(renderer='plotly_mimetype+notebook')
