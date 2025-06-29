from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .inference import predict
from .schemas import ApplicantRequest, PredictionResponse

# Inicializar la estancia de la app
app = FastAPI(
    title="Good/Bad Applicant Prediction API",
    description="An API for predicting whether a loan applicant is good or bad.",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Health check endpoint
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: ApplicantRequest):
    """
    Toma como entrada una estructura json con las features
    necesarias para la predicción.
    Devuelve una diccionario con las siguientes llaves:
    1. prediction: 0 si es BUENO, 1 si es MALO.
    2. probability: Probabilidad que el aplicante sea considerado como "malo".
    """
    return predict(request)
