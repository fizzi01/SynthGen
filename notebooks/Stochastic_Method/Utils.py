from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go



class Category(Enum):
    SC1 = 1
    SC2 = 2
    SC3 = 3
    SC4 = 4
    SC5 = 5


def day_category(valore: float) -> Category:
    """
    Restituisce la categoria di appartenenza in base al valore fornito.

    :param valore: Il valore numerico da classificare.
    :return: Una stringa rappresentante la categoria di appartenenza.
    """
    if 0 <= valore <= 0.2:
        return Category.SC1
    elif 0.2 < valore <= 0.4:
        return Category.SC2
    elif 0.4 < valore <= 0.6:
        return Category.SC3
    elif 0.6 < valore <= 0.67:
        return Category.SC4
    elif 0.67 < valore <= 1:
        return Category.SC5
    else:
        if valore < 0:
            return Category.SC1
        else:
            return Category.SC5


def day_categorizer(dati: list, costanti_dict: dict) -> dict[Category, list[str]]:
    """
    Calcola la media dei valori per ogni giorno dopo aver diviso ciascun valore
    per la sua corrispondente costante giornaliera, e assegna a ogni giorno una categoria SC.

    :param dati: Lista di dizionari contenenti i valori e i rispettivi timestamp.
    :param costanti_dict: Lista di costanti, una per ogni giorno.
    :return: Un dizionario con categorie come chiavi, e come valore i singoli giorni.
    """
    # Assumendo che dati sia già un DataFrame per semplicità
    df = pd.DataFrame(dati)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['data'] = df['timestamp'].dt.date

    # Applicare la costante corrispondente utilizzando la seconda colonna del DataFrame
    colonna_valori = df.columns[1]  # Nome della seconda colonna che contiene i valori
    df['costants'] = pd.DataFrame(list(costanti_dict.values()))
    df['valore_normalizzato'] = df[colonna_valori] / df['costants']
    df['valore_normalizzato'] = df['valore_normalizzato'].fillna(0)
    df.loc[df['valore_normalizzato'] == float("inf"), 'valore_normalizzato'] = 0
    df.loc[df['valore_normalizzato'] > 1, 'valore_normalizzato'] = 1
    df = df[df['valore_normalizzato'] != 0]

    # Calcolare la media giornaliera dei valori normalizzati
    media_giornaliera = df.groupby('data')['valore_normalizzato'].mean().reset_index()

    # Categorizzare ciascuna media giornaliera
    risultati = {}
    for _, riga in media_giornaliera.iterrows():
        categoria = day_category(riga['valore_normalizzato'])
        risultati[str(riga['data'])] = categoria

    return group_by_category(risultati)


def group_by_category(date_categorie: dict[str, Category]) -> dict[Category, list[str]]:
    """
    Raggruppa i giorni per categoria.

    :param date_categorie: Dizionario con date (come stringhe) e categorie (come membri di Category).
    :return: Dizionario con categorie come chiavi e liste di date come valori.
    """
    risultato = {categoria: [] for categoria in Category}  # Inizializza il dizionario con tutte le categorie possibili

    for data, categoria in date_categorie.items():
        risultato[categoria].append(data)

    # Opzionale: ordinare le date all'interno di ogni categoria
    for categoria in risultato:
        risultato[categoria].sort()

    return risultato


def plot_dataframes(dataframes, title: str = "Plot"):
    """
    Fa il plot di una lista di DataFrame su un unico grafico con colori diversi.

    :param title: Titolo dataframe
    :param dataframes: Lista di DataFrame da plottare.
    """

    plt.figure(figsize=(24, 6))  # Imposta la dimensione del grafico

    for df in dataframes:
        df.plot()
        plt.xlabel('Timestamp')
        plt.ylabel('Valore')
        plt.title(title)
        plt.xticks(rotation=90)  # Ruota le etichette dell'asse x per migliorare la leggibilità
        plt.legend()
        plt.tight_layout()
        plt.grid(True)  # Aggiunge una griglia per migliorare la leggibilità
        plt.show()


"""def lets_plot_dataframes(dataframes, title: str = "Plot"):
    
    Fa il plot di una lista di DataFrame su un unico grafico con colori diversi.

    :param title: Titolo dataframe
    :param dataframes: Lista di DataFrame da plottare.
    

    # Trasforma i DataFrame in un formato long per il plotting
    for df in dataframes:
        # "Sciogliere" il DataFrame da un formato wide a un formato long
        df_long = df.reset_index().melt(id_vars=['index'], var_name='Parameter', value_name='Value')

        # Rinominare la colonna 'index' per chiarezza
        df_long.rename(columns={'index': 'Timestamp'}, inplace=True)

        # Creare il plot
        p = ggplot(df_long, aes(x='Timestamp', y='Value', color='Parameter')) + geom_line() + theme_minimal() + ggsize(
            800, 400)

        p += ggtitle(title)
        p.show()"""


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

