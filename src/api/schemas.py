from pydantic import BaseModel, Field
from typing import List

class ApplicantRequest(BaseModel):
    FLAG_OWN_CAR: str = Field(..., description="Flag for car ownership")
    FLAG_OWN_REALTY: str = Field(..., description="Flag for real estate ownership")
    CNT_CHILDREN: int = Field(..., ge=0, description="Number of children")
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Total income amount")
    NAME_INCOME_TYPE: str = Field(..., description="Type of income")
    NAME_EDUCATION_TYPE: str = Field(..., description="Level of education")
    NAME_FAMILY_STATUS: str = Field(..., description="Marital status")
    NAME_HOUSING_TYPE: str = Field(..., description="Housing situation")
    DAYS_BIRTH: int = Field(..., description="Days from birth")
    DAYS_EMPLOYED: int = Field(..., description="Days employed")
    FLAG_WORK_PHONE: int = Field(..., description="Work phone flag")
    FLAG_PHONE: int = Field(..., description="Phone flag")
    FLAG_EMAIL: int = Field(..., description="Email flag")
    OCCUPATION_TYPE: str = Field(..., description="Occupation")

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
