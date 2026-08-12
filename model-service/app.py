from fastapi import FastAPI

app = FastAPI(title="Model Service")

@app.get("/health")
def health():
    return {"status": "ok"}
