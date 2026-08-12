from fastapi import APIRouter

router = APIRouter()

@router.get("/agent")
def agent_status():
    return {"status": "ready"}
