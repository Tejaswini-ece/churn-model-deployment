
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    @field_validator(
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    )
    @classmethod
    def validate_category(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Category cannot be empty")
        return value


class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_churn: bool
    threshold: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerInput] = Field(min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

