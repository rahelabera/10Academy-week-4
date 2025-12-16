from pydantic import BaseModel

class CustomerInput(BaseModel):
    # Define key raw features if predicting on raw, or assume preprocessed
    pass  # Or list aggregates

class PredictionOutput(BaseModel):
    risk_probability: float
    is_high_risk: int