from pydantic import BaseModel

class Transaction(BaseModel):
    amount: float
    feature_vector: list[float]
