import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.api.inference import predict
from src.api.schemas import ApplicantRequest, PredictionResponse

# Mock para el modelo y la configuración
@pytest.fixture
def mock_model_and_config():
    with patch('src.api.inference.model') as mock_model:
        with patch('src.api.inference.get_config') as mock_get_config:
            # Mock del modelo
            mock_model.predict_proba.return_value = [[0.7, 0.3]] # Ejemplo: 30% de probabilidad de ser malo

            # Mock de la configuración
            mock_config_data = {
            "model": {
                "list_numeric_features": ["AGE", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "WORK_YEARS"],
                "list_categorical_features": [
                    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
                    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
                    "OCCUPATION_TYPE", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"
                ]
            }
        }
        mock_get_config.return_value = mock_config_data

        yield mock_model, mock_config_data

def test_predict_good_applicant(mock_model_and_config):
    mock_model, mock_config_data = mock_model_and_config

    request_data = ApplicantRequest(
        FLAG_OWN_CAR="Y",
        FLAG_OWN_REALTY="Y",
        CNT_CHILDREN=0,
        AMT_INCOME_TOTAL=216000.0,
        NAME_INCOME_TYPE="Working",
        NAME_EDUCATION_TYPE="Higher education",
        NAME_FAMILY_STATUS="Married",
        NAME_HOUSING_TYPE="House / apartment",
        DAYS_BIRTH=-18529,
        DAYS_EMPLOYED=-1809,
        FLAG_WORK_PHONE=0,
        FLAG_PHONE=0,
        FLAG_EMAIL=0,
        OCCUPATION_TYPE="Secretaries"
    )

    response = predict(request_data)

    assert isinstance(response, PredictionResponse)
    assert response.prediction == "BUENO"  # 0.3 probability, so good applicant
    assert response.probability == 0.3

    # Verificar que predict_proba fue llamado con un DataFrame de pandas
    mock_model.predict_proba.assert_called_once()
    args, kwargs = mock_model.predict_proba.call_args
    assert isinstance(args[0], pd.DataFrame)
    assert not args[0].empty

def test_predict_bad_applicant(mock_model_and_config):
    mock_model, mock_config_data = mock_model_and_config

    # Cambiar el valor de retorno para simular un "mal" aplicante
    mock_model.predict_proba.return_value = [[0.2, 0.8]] # Ejemplo: 80% de probabilidad de ser malo

    request_data = ApplicantRequest(
        FLAG_OWN_CAR="N",
        FLAG_OWN_REALTY="N",
        CNT_CHILDREN=2,
        AMT_INCOME_TOTAL=50000.0,
        NAME_INCOME_TYPE="Unemployed",
        NAME_EDUCATION_TYPE="Lower secondary",
        NAME_FAMILY_STATUS="Single / not married",
        NAME_HOUSING_TYPE="Rented apartment",
        DAYS_BIRTH=-10000,
        DAYS_EMPLOYED=-500,
        FLAG_WORK_PHONE=1,
        FLAG_PHONE=1,
        FLAG_EMAIL=1,
        OCCUPATION_TYPE="Low-skill Laborers"
    )

    response = predict(request_data)

    assert isinstance(response, PredictionResponse)
    assert response.prediction == "MALO"  # 0.8 probability, so bad applicant
    assert response.probability == 0.8
