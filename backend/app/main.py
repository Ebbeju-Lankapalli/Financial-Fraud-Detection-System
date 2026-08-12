from fastapi import FastAPI

app = FastAPI(title="Financial Fraud Detection System")

@app.get("/")
def read_root():
    return {"message": "Financial Fraud Detection System API"}
