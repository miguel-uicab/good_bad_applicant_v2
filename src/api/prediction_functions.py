# !/usr/bin/env python
# coding: utf-8

import yaml
import pandas as pd


def get_config(path: str = None):
    """
    Se carga el archivo config.yaml.

    Args:
        path (str, optional): Ruta al archivo de configuración. 

    Returns:
        dict: Un diccionario con la configuración del modelo.
    """
    if path is None:
        path = './configs/model_config.yaml'

    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def get_dataframe(json: dict = None) -> pd.DataFrame:
    """
    Convierte una estructura json con la informarción básica payload
    en un DataFrame. El .T rota el DataFrame con el fin de obtener
    información tipo fila.

    Args:
        json (dict, optional): Un diccionario con los datos de entrada. 

    Returns:
        pd.DataFrame: Un DataFrame de pandas con la información del JSON.
    """
    df_json = pd.DataFrame.from_dict(json, orient='index').T

    return df_json


def fillna_categoric_data(data: pd.DataFrame = None,
                          list_names: list = None) -> pd.DataFrame:
    """
    Dada una lista de variables categóricas, imputa, en los
    valores perdidos, el valor de 'Unidentified'.

    Args:
        data (pd.DataFrame, optional): El DataFrame a procesar.
        list_names (list, optional): Lista de columnas categóricas a rellenar. 

    Returns:
        pd.DataFrame: El DataFrame con los valores faltantes imputados.
    """
    data_copy = data.copy()
    for name in list_names:
        data_copy[name].fillna('Unidentified', inplace=True)

    return data_copy


def get_feature_names_order(float_names: list = None,
                            categorical_names: list = None) -> list:
    """
    El uso de Pipilines implica que el orden de las variables importa.
    Este orden en las variables esta vínculado al orden en que suceden los
    procesos en las tuberías (ver función "get_preprocessor"). En este contexto,
    el orden de aparición de la variables queda como sigue:
    1. Tipo categórica.
    2. Tipo float.

    Args:
        float_names (list, optional): Lista de nombres de variables float.
        categorical_names (list, optional): Lista de nombres de variables categóricas. 

    Returns:
        list: Una lista con el orden correcto de las variables.
    """
    feature_names_order = categorical_names + float_names

    return feature_names_order


def transform_data_type_to_float(data: pd.DataFrame = None,
                                 list_names: list = None) -> pd.DataFrame:
    """
    Dada una lista de variables de interés, se tranforman
    a tipo float.

    Args:
        data (pd.DataFrame, optional): El DataFrame a procesar. 
        list_names (list, optional): Lista de columnas a transformar a float. 

    Returns:
        pd.DataFrame: El DataFrame con las columnas transformadas a tipo float.
    """
    data_copy = data.copy()
    for name in list_names:
        data_copy[name] = data_copy[name].astype(float)

    return data_copy


def NAME_EDUCATION_TYPE_class(education_type: str) -> str:
    """
    Tiene como tarea homogeneizar los valores de la variable "NAME_EDUCATION_TYPE".

    Args:
        education_type (str): El tipo de educación a clasificar.

    Returns:
        str: El valor homogeneizado del tipo de educación.
    """
    if education_type in ['Higher education', 'Academic degree']:
        return 'Higher education or Academic degree'
    elif education_type in ['Lower secondary', 'Incomplete higher']:
        return 'Lower secondary or Incomplete higher'
    else:
        return education_type


def NAME_HOUSING_TYPE_clas(housing_type: str) -> str:
    """
    Tiene como tarea homogeneizar los valores de la variable "NAME_HOUSING_TYPE".

    Args:
        housing_type (str): El tipo de vivienda a clasificar.

    Returns:
        str: El valor homogeneizado del tipo de vivienda.
    """
    if housing_type in ['Rented apartment', 'Office apartment', 'Co-op apartment']:
        return 'Rented apartment or Office apartment or Co-op apartment'
    else:
        return housing_type


def OCCUPATION_TYPE_class(occupation_type: str) -> str:
    """
    Tiene como tarea homogeneizar los valores de la variable "OCCUPATION_TYPE".

    Args:
        occupation_type (str): El tipo de ocupación a clasificar.

    Returns:
        str: El valor homogeneizado del tipo de ocupación.
    """
    if occupation_type in ['Cleaning staff', 'Private service staff', 'Secretaries',
                           'Waiters/barmen staff', 'Low-skill Laborers', 'IT staff',
                           'Realty agents', 'HR staff']:
        return 'Others'
    else:
        return occupation_type
