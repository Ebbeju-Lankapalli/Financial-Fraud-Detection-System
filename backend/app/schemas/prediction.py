from pydantic import BaseModel

class Prediction(BaseModel):
    risk_score: float
    label: str
