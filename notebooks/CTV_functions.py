# !/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import plotly.express as px


def threshold_str(due_list=None):
    """
    Proporciona un string necesario para el titulo de las gráficas.

    Args:
        due_list (list): Lista de strings que representa las categorías de mora.

    Returns:
        str: Un string que representa el mínimo de días de atraso (e.g., '30', '60').
    """
    if ('1' in due_list) and (len(due_list) == 5):
        string = '30'
    elif ('2' in due_list) and (len(due_list) == 4):
        string = '60'
    elif ('3' in due_list) and (len(due_list) == 3):
        string = '90'
    elif ('4' in due_list) and (len(due_list) == 2):
        string = '120'
    elif ('5' in due_list) and (len(due_list) == 1):
        string = '150'

    return string


def filter_integers(interval=None,
                    lst=None):
    """
    Filtra una lista de enteros basada en un intervalo de tiempo.

    Las opciones para el intervalo son:
    1. "bimonthly": por bimestre.
    2. "quarterly": por trimestre.
    3. "four-month period": por cuatrimestre.
    4. "semester": por semestre.
    5. "yearly": anual.

    Args:
        interval (str): El intervalo de tiempo para filtrar.
        lst (list): Una lista de enteros para ser filtrada.

    Returns:
        list: Una lista ordenada y filtrada de enteros.
    """
    sorted_lst = sorted(lst, reverse=True)
    if interval == "bimonthly":
        return sorted_lst[::2]
    elif interval == "quarterly":
        return sorted_lst[::3]
    elif interval == "four-month period":
        return sorted_lst[::4]
    elif interval == "semester":
        return sorted_lst[::6]
    elif interval == "yearly":
        return sorted_lst[::12]
    else:
        raise ValueError("Interval not recognized")


def get_cohort_graph(df_vintage=None,
                     chosen_bucket=None):
    """
    Construye y muestra el gráfico de las cohorts.

    Args:
        df_vintage (pd.DataFrame): DataFrame que contiene los datos del análisis de cohorts.
        chosen_bucket (list): Lista de strings con las categorías de mora para determinar el título.
    
    Returns:
        None: Muestra una figura de Plotly.
    """
    threshold_value = threshold_str(chosen_bucket)
    fig = px.line(df_vintage,
                  x='month_on_book',
                  y='due_rate',
                  color='opening_month',
                  markers=True)
    fig.update_traces(marker=dict(size=5))
    title_str = f"Porcentaje acumulado de clientes morosos (> {threshold_value} días de atraso)"
    fig.update_layout(title=title_str)
    fig.show()


def get_vintage_analysis(n=None,
                         df_credit_extend=None,
                         chosen_bucket=None):
    """
    Obtención de la tabla que contiene el Análisis de cohorts.
    Aquí n se refiere a un entero positivo con el que se filtrarán
    las observaciones cuyas ventanas de observación (window)
    sean menores o igual a n.

    Args:
        n (int): Ventana de observación mínima para incluir un ID.
        df_credit_extend (pd.DataFrame): El dataset de crédito extendido.
        chosen_bucket (list): Lista de strings que representan los status de "mal solicitante".

    Returns:
        tuple: Una tupla conteniendo:
            - df_vintage (pd.DataFrame): DataFrame con el análisis de cohorts.
            - due_ids_in_cohorts (list): Una lista de listas con los IDs de los clientes "malos" por cohort.
    """
    # Se eliminan los ids cuya ventana de observación se menor que n.
    df_credit_extend_trunc = df_credit_extend[df_credit_extend['window'] >= n]

    # Etiquedado de las "malos clientes".
    df_credit_extend_trunc['STATUS_0_1'] = 0
    bad_aplicants_index_list = (df_credit_extend_trunc['STATUS'].isin(chosen_bucket))
    df_credit_extend_trunc.loc[bad_aplicants_index_list, 'STATUS_0_1'] = 1

    # Cálculo de la frecuencia en cada cohort.
    df_count_by_cohort = (df_credit_extend_trunc.groupby(['opening_month'])['ID']
                                                .nunique()
                                                .to_frame())
    df_count_by_cohort.reset_index(inplace=True)
    df_count_by_cohort.columns = ['opening_month', 'cohort_count']

    # Comienza la construcción de la tabla que contendrá
    # el Análisis de Cohorts.
    new_columns = ['opening_month', 'month_on_book']
    df_raw_vintage = (df_credit_extend_trunc[new_columns].drop_duplicates()
                                                         .sort_values(new_columns)
                                                         .reset_index(drop=True))
    df_raw_vintage['due_count'] = np.nan

    # Se una cada cohort con su frecuencia.
    df_vintage = pd.merge(df_raw_vintage,
                          df_count_by_cohort,
                          on=['opening_month'],
                          how='left')

    # Cálculo de la frecuencia de "clientes malos" en cada cohort.
    due_ids_in_cohorts = []  # Guardado de los ids de "malos clientes" en cada cohort.
    for i in range(-60, 1):  # Mes en el que se abrió la cuenta (opening_month).
        due_ids_lst = []  # Guardado de las freciencias para el mes de inicio i.
        for j in range(0, 61):  # Mes después de que se abrió la cuenta (month_on_book).
            df_1 = df_credit_extend_trunc[(df_credit_extend_trunc['STATUS_0_1']==1)  # localización de los clientes morosos.
                                          & (df_credit_extend_trunc['month_on_book'] == j)
                                          & (df_credit_extend_trunc['opening_month'] == i)]
            due_ids = list(df_1['ID'])  # Se obtienen los ids de los "malos clientes" para opening_month i y el month_ob_book j.
            due_ids_lst.extend(due_ids)  # Se añaden los "malos clientes" a medida que que pasan los meses.
            df_vintage.loc[(df_vintage['month_on_book'] == j)
                            & (df_vintage['opening_month'] == i), 'due_count'] = len(set(due_ids_lst))  # Se añade la freciencia, profurando que no haya duplicados.
        due_ids_in_cohorts.append(due_ids_lst)  # Se añaden los "clientes malos" a medida que cambia el mes de inicio.
 
    # Cálculo de la frecuenca de "malos clientes" para cada month_on_book
    # Pertenciente a determinando cohort.
    df_vintage['due_rate'] = df_vintage['due_count'] / df_vintage['cohort_count']  # calcular el % acumulado de clientes morosos"

    df_vintage['due_rate'] = round(df_vintage['due_rate'] * 100, 2)

    return df_vintage, due_ids_in_cohorts


def get_vintage_analysis_by_interval(df_vintage=None,
                                     time_interval=None,
                                     cohorts_list=None):
    """
    Hace un filtrado del análisis completo de cohorts usando
    intervalos de tiempo o una lista de cohorts.

    Depende de la la tabla resultante de la función "get_vintage_analysis".
    Las opciones para time_interval son:
    1. "bimonthly": por bimestre.
    2. "quarterly": por trimestre.
    3. "four-month period": por cuatrimestre.
    4. "semester": por semestre.
    5. "yearly": anual.
    
    Args:
        df_vintage (pd.DataFrame): El resultado de `get_vintage_analysis`.
        time_interval (str, optional): El intervalo de tiempo para filtrar.
        cohorts_list (list, optional): Una lista específica de cohorts para filtrar.

    Returns:
        pd.DataFrame: El DataFrame del análisis de cohorts filtrado.
    """
    if time_interval:
        lst_opening_month = df_vintage['opening_month'].unique().tolist()
        list_period = filter_integers(time_interval, lst_opening_month)
        df_final = df_vintage[df_vintage['opening_month'].isin(list_period)]
    else:
        df_final = df_vintage[df_vintage['opening_month'].isin(cohorts_list)]

    return df_final


def get_mean_cohort(df_vintage=None,
                    chosen_bucket=None,
                    with_graph=True):
    """
    Cálcula el "cohort medio" a través del promedio
    de las frecuencias acumuladas vistas en
    cada month_on_book.

    Args:
        df_vintage (pd.DataFrame): Los datos del análisis de cohorts.
        chosen_bucket (list): Lista de strings con las categorías de mora para el título.
        with_graph (bool): Si es True, muestra un gráfico.

    Returns:
        pd.DataFrame: Un DataFrame con los porcentajes del cohort promedio.
    """
    threshold_value = threshold_str(chosen_bucket)
    df_vintage_pivot = df_vintage.pivot(index='opening_month',
                                        columns='month_on_book',
                                        values='due_rate')

    d_mean = df_vintage_pivot.reset_index(drop=True).mean().to_frame()
    d_mean.reset_index(drop=False, inplace=True)
    d_mean.columns = ['month_on_book', 'mean_porcentajes']

    graph_tittle = f'Cohort promedio (> {threshold_value} días de atraso)'
    if with_graph:
        fig = px.scatter(d_mean,
                         x='month_on_book',
                         y='mean_porcentajes',
                         title=graph_tittle)
        line_fig = px.line(d_mean,
                           x='month_on_book',
                           y='mean_porcentajes',
                           line_shape='linear')

        fig.add_traces(line_fig.data)

        fig.show()

    return d_mean


def inner_join(df1, df2):
    """
    Realiza un inner join en dos DataFrames.

    Args:
        df1 (pd.DataFrame): El primer DataFrame.
        df2 (pd.DataFrame): El segundo DataFrame.

    Returns:
        pd.DataFrame: El DataFrame resultado del merge.
    """
    df = pd.merge(df1,
                  df2,
                  on='month_on_book',
                  how='inner',
                  suffixes=('_df1', '_df2'))

    return df


# NOTA: La construcción de estas funciones, sobretodo de
# la función "get_vintage_analysis" están fuertemente inspirada
# en un Análisis de Cohorts hecho para esta misma data.
# Se puede encontrar aquí:
# https://www.kaggle.com/code/rikdifos/eda-vintage-analysis
# Algunas consideraciones de código se han tomado de ahí, otras se han cambiado
# totalmente, y otras se han anexado.
# Nunca he hecho un Análisis de Cohorts, así que después de documentarme,
# buscar y correr algunos ejemplos en Medium, me tope con ella en Kaggle.
