```bash
cd deployment/mlflow
docker compose -f docker-compose.yaml up -d
docker compose ps
```


```bash
python src/models/train_model.py   --config configs/model_config.yaml   --data data/processed/df_cleaned_featured.sav   --models-dir models   --mlflow-tracking-uri http://localhost:5555
```

### Prueba de la API

Para probar la API localmente, primero asegúrate de que la aplicación FastAPI esté corriendo (por ejemplo, usando `uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000`). Luego, puedes enviar una solicitud POST al endpoint `/predict` con el siguiente payload:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
    "FLAG_OWN_CAR": "Y",
    "FLAG_OWN_REALTY": "Y",
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 216000.0,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "DAYS_BIRTH": -18529,
    "DAYS_EMPLOYED": -1809,
    "FLAG_WORK_PHONE": 0,
    "FLAG_PHONE": 0,
    "FLAG_EMAIL": 0,
    "OCCUPATION_TYPE": "Secretaries"
}'
```

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

```bash
docker image build -t migueluicab/good-bad-applicant-api:v1 -f Dockerfile.api .
```

```bash
 docker run -p 8000:8000 migueluicab/good-bad-applicant-api:v1
 ```

### Ejecución de Pruebas Unitarias

Para ejecutar las pruebas unitarias del proyecto, asegúrate de tener el ambiente virtual activado y `pytest` instalado. Luego, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
uv run pytest tests/api
```

### Ejecución de la Aplicación Streamlit

Para levantar la aplicación Streamlit, asegúrate de tener el ambiente virtual activado y `streamlit` instalado. Luego, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
uv run streamlit run streamlit_app/app.py
```

```bash
docker build -t migueluicab/good-bad-applicant-streamlit:v1 -f Dockerfile.streamlit .
```



```bash
docker run -p 8501:8501 migueluicab/good-bad-applicant-streamlit:v1
```

```bash
http://localhost:8501/
```

```bash
docker-compose up --build
```

```bash
docker-compose up
```

```bash
docker-compose down
```

```bash
docker image push migueluicab/good-bad-applicant-streamlit:latest
```

```bash
docker image push migueluicab/good-bad-applicant-api:latest
```