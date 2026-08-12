from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "financial-fraud-detection-system"

settings = Settings()
