from pydantic import BaseModel


class AgentMessage(BaseModel):
    message: str
