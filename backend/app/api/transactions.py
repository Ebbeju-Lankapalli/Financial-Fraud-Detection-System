from fastapi import APIRouter

router = APIRouter()

@router.get("/transactions")
def list_transactions():
    return []
