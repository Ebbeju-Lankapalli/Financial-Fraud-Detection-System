FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY model-service/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY model-service /app/model-service

COPY artifacts/ieee_cis/catboost_reduced_fraud_detector.cbm \
     /app/artifacts/ieee_cis/catboost_reduced_fraud_detector.cbm

WORKDIR /app/model-service

CMD ["sh", "-c", "uvicorn production_app:app --host 0.0.0.0 --port ${PORT:-8001}"]
