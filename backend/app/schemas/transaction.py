from pydantic import BaseModel

class Transaction(BaseModel):
    amount: float
    currency: str
