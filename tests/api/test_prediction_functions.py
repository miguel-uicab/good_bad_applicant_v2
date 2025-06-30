import pandas as pd
import pytest
from src.api.prediction_functions import (
    get_config,
    get_dataframe,
    fillna_categoric_data,
    transform_data_type_to_float,
    NAME_EDUCATION_TYPE_class,
    NAME_HOUSING_TYPE_clas,
    OCCUPATION_TYPE_class,
)

# Mock para get_config (ya que depende de un archivo)
@pytest.fixture
def mock_config(tmp_path):
    config_content = """
model:
  list_numeric_features:
  - AGE
  - CNT_CHILDREN
  - AMT_INCOME_TOTAL
  - WORK_YEARS
  list_categorical_features:
  - FLAG_OWN_CAR
  - FLAG_OWN_REALTY
  - NAME_INCOME_TYPE
  - NAME_EDUCATION_TYPE
  - NAME_FAMILY_STATUS
  - NAME_HOUSING_TYPE
  - OCCUPATION_TYPE
  - FLAG_WORK_PHONE
  - FLAG_PHONE
  - FLAG_EMAIL
"""
    config_file = tmp_path / "model_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)

def test_get_config(mock_config):
    config = get_config(mock_config)
    assert "model" in config
    assert "list_numeric_features" in config["model"]
    assert "list_categorical_features" in config["model"]

def test_get_dataframe():
    json_data = {"col1": 1, "col2": "test"}
    df = get_dataframe(json_data)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 2)
    assert df["col1"].iloc[0] == 1

def test_fillna_categoric_data():
    df = pd.DataFrame({
        "cat_col": ["A", None, "B"],
        "num_col": [1, 2, 3]
    })
    list_names = ["cat_col"]
    result_df = fillna_categoric_data(data=df, list_names=list_names)
    assert result_df["cat_col"].iloc[1] == "Unidentified"
    assert result_df["num_col"].iloc[1] == 2 # Asegura que otras columnas no cambian

def test_transform_data_type_to_float():
    df = pd.DataFrame({"col1": ["1.0", "2.5"], "col2": ["A", "B"]})
    list_names = ["col1"]
    result_df = transform_data_type_to_float(data=df, list_names=list_names)
    assert result_df["col1"].dtype == float
    assert result_df["col1"].iloc[0] == 1.0

def test_NAME_EDUCATION_TYPE_class():
    assert NAME_EDUCATION_TYPE_class("Higher education") == "Higher education or Academic degree"
    assert NAME_EDUCATION_TYPE_class("Academic degree") == "Higher education or Academic degree"
    assert NAME_EDUCATION_TYPE_class("Lower secondary") == "Lower secondary or Incomplete higher"
    assert NAME_EDUCATION_TYPE_class("Incomplete higher") == "Lower secondary or Incomplete higher"
    assert NAME_EDUCATION_TYPE_class("Secondary / secondary special") == "Secondary / secondary special"

def test_NAME_HOUSING_TYPE_clas():
    assert NAME_HOUSING_TYPE_clas("Rented apartment") == "Rented apartment or Office apartment or Co-op apartment"
    assert NAME_HOUSING_TYPE_clas("Office apartment") == "Rented apartment or Office apartment or Co-op apartment"
    assert NAME_HOUSING_TYPE_clas("Co-op apartment") == "Rented apartment or Office apartment or Co-op apartment"
    assert NAME_HOUSING_TYPE_clas("House / apartment") == "House / apartment"

def test_OCCUPATION_TYPE_class():
    assert OCCUPATION_TYPE_class("Secretaries") == "Others"
    assert OCCUPATION_TYPE_class("IT staff") == "Others"
    assert OCCUPATION_TYPE_class("Managers") == "Managers"
