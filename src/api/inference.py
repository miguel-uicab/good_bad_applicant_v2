import os
import pickle
import pandas as pd
from pathlib import Path
from .prediction_functions import (
    get_config,
    get_dataframe,
    fillna_categoric_data,
    transform_data_type_to_float,
    NAME_EDUCATION_TYPE_class,
    NAME_HOUSING_TYPE_clas,
    OCCUPATION_TYPE_class,
)
from .schemas import ApplicantRequest, PredictionResponse


# Determinar la ruta base para poder cargar archivos tanto en local como en Docker
# Si la variable de entorno 'DOCKER_ENV' se establece en 'true', usamos la ruta de Docker.
de_build = os.environ.get('DE_BUILD', 'local') == 'docker'
base_path = Path('/app') if de_build else Path('.')

# Cargar modelo y configuraciones
MODEL_PATH = base_path / "models" / "trained" / "good_bad_applicant_model.pkl"
CONFIG_PATH = base_path / "configs" / "model_config.yaml"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    config = get_config(CONFIG_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or config: {str(e)}")

def predict(request: ApplicantRequest) -> PredictionResponse:
    """
    Realiza la predicción de si un aplicante es bueno o malo.
    """
    # features_names = config['list_features_names']
    list_numeric_names = config['model']['list_numeric_features']
    list_categorical_names = config['model']['list_categorical_features']
    # variable_names = config['variable_names']

    # Crear DataFrame a partir de la solicitud
    input_data = pd.DataFrame([request.dict()])

    # Transformaciones de datos
    df = transform_data_type_to_float(data=input_data,
                                      list_names=['DAYS_BIRTH',
                                                  'DAYS_EMPLOYED'])

    df["AGE"] = -df['DAYS_BIRTH'] / 365
    df["AGE"] = df["AGE"].astype(int)
    df['WORK_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    df["WORK_YEARS"] = df["WORK_YEARS"].astype(int)

    df = transform_data_type_to_float(data=df, list_names=list_numeric_names)

    # Tratamiento de variables categóricas
    df_FILLNA_CATEGORIC = fillna_categoric_data(data=df, list_names=list_categorical_names)

    df_HOMO_CLASS = df_FILLNA_CATEGORIC.copy()

    df_HOMO_CLASS['NAME_EDUCATION_TYPE'] = list(map(NAME_EDUCATION_TYPE_class, df_HOMO_CLASS['NAME_EDUCATION_TYPE']))
    df_HOMO_CLASS['NAME_HOUSING_TYPE'] = list(map(NAME_HOUSING_TYPE_clas, df_HOMO_CLASS['NAME_HOUSING_TYPE']))
    df_HOMO_CLASS['OCCUPATION_TYPE'] = list(map(OCCUPATION_TYPE_class, df_HOMO_CLASS['OCCUPATION_TYPE']))

    # Ordenar las columnas
    # feature_names_order = get_feature_names_order(float_names=float_names, categorical_names=categorical_names)
    data_order = df_HOMO_CLASS[list_numeric_names + list_categorical_names]

    # Realizar predicción
    predict_array = model.predict_proba(data_order)
    probability = round(predict_array[0][1], 2)

    # Asignar categoría
    prediction = 'MALO' if probability > 0.5 else 'BUENO'

    return PredictionResponse(prediction=prediction, probability=probability)
