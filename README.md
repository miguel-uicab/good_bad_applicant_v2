# Good/Bad Applicant Classifier

Este proyecto es una solución de Machine Learning de extremo a extremo diseñada para clasificar a los solicitantes de crédito como "buenos" o "malos" en función de su historial financiero y de aplicación. El objetivo es automatizar y mejorar la precisión del proceso de aprobación de créditos.

La solución incluye una API de FastAPI para servir las predicciones del modelo y una aplicación web interactiva de Streamlit para una fácil utilización por parte de los usuarios finales.

## 📋 Características Principales

*   **Modelo de Clasificación:** Utiliza un modelo `HistogramGradientBoosting` para predecir la probabilidad de que un solicitante sea un buen o mal cliente.
*   **API RESTful:** Una API robusta construida con **FastAPI** para servir predicciones en tiempo real. Incluye documentación interactiva con Swagger UI.
*   **Interfaz de Usuario Web:** Una aplicación intuitiva desarrollada con **Streamlit** que permite a los usuarios introducir datos de un solicitante y recibir una clasificación al instante.
*   **Contenerización:** Todo el proyecto está contenerizado con **Docker**, permitiendo una configuración y despliegue sencillos y reproducibles.
*   **Pipeline de MLOps Automatizado:** Un pipeline de CI/CD con **GitHub Actions** que automáticamente prueba el código, entrena el modelo, lo registra con **MLflow** y publica las imágenes de la API y la aplicación de Streamlit en Docker Hub.

## 🛠️ Pila Tecnológica

*   **Backend:** Python, FastAPI
*   **Frontend:** Streamlit
*   **Machine Learning:** Scikit-learn, Xgboost, Pandas, MLflow
*   **Despliegue:** Docker, Docker Compose
*   **CI/CD:** GitHub Actions

## 📂 Estructura del Proyecto

```
good_bad_applicant_v2/
├── .github/workflows/mlops-pipeline.yaml  # Pipeline de CI/CD
├── data/                                  # Datos crudos y procesados
├── models/                                # Modelo entrenado
├── notebooks/                             # Jupyter Notebooks para análisis y experimentación
├── src/
│   ├── api/                               # Código fuente de la API (FastAPI)
│   └── models/                            # Scripts para entrenamiento de modelos
├── streamlit_app/                         # Código fuente de la App (Streamlit)
├── tests/                                 # Pruebas unitarias y de integración
├── docker-compose.yaml                    # Orquestador de servicios locales
├── Dockerfile.api                         # Dockerfile para la API
├── Dockerfile.streamlit                   # Dockerfile para la App de Streamlit
└── README.md                              # Este archivo
```

## 🚀 Cómo Empezar

Para ejecutar este proyecto localmente, asegúrate de tener **Docker** y **Docker Compose** instalados.

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/your-username/good_bad_applicant_v2.git
    cd good_bad_applicant_v2
    ```

2. **Crea el ambiente virtual de python usando UV:**
   Si se quiere correr los notebooks en local, hay que crear el ambiente,
   ```bash
   uv venv --python=python3.11
   source .venv/bin/activate
   ```
   e instalar las dependencias,
   
   ```bash
   uv pip install -r requirements.txt
   ```

3.  **Levanta los servicios con Docker Compose:**
    Este comando construirá las imágenes y levantará los contenedores para la API y la aplicación de Streamlit.
    ```bash
    docker-compose up --build
    ```

4.  **¡Listo! Accede a los servicios:**
    *   **API (FastAPI):** Abre tu navegador y ve a `http://localhost:8000`.
    *   **Documentación de la API:** Para interactuar con la API, ve a `http://localhost:8000/docs`.
    *   **Aplicación (Streamlit):** Abre tu navegador y ve a `http://localhost:8501`.

## ⚙️ Pipeline de MLOps

Este proyecto utiliza un pipeline automatizado de MLOps definido en `.github/workflows/mlops-pipeline.yaml`. Este flujo de trabajo se activa con cada `push` o `pull request` a la rama `main` y consta de los siguientes jobs:

1.  **`test`**:
    *   Instala las dependencias.
    *   Ejecuta las pruebas con `pytest` para garantizar la calidad y estabilidad del código.

2.  **`train-and-register-model`**:
    *   Si las pruebas pasan, este job entrena el modelo de clasificación.
    *   Inicia un servidor de **MLflow** temporal para registrar el experimento, las métricas y el modelo entrenado.
    *   Guarda el modelo (`good_bad_applicant_model.pkl`) como un artefacto para el siguiente paso.

3.  **`build-and-publish-api` y `build-and-publish-streamlit`**:
    *   Estos jobs se ejecutan en paralelo una vez que el modelo ha sido entrenado.
    *   **`build-and-publish-api`**: Descarga el artefacto del modelo, construye la imagen Docker de la API usando `Dockerfile.api` y la publica en Docker Hub.
    *   **`build-and-publish-streamlit`**: Construye la imagen Docker de la aplicación de Streamlit usando `Dockerfile.streamlit` y la publica en Docker Hub.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si deseas mejorar este proyecto, por favor haz un fork del repositorio y crea un Pull Request.

