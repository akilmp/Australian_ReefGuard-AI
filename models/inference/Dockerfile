# Simple Dockerfile for KFServing Predictor
FROM python:3.9-slim

WORKDIR /app

# Copy predictor implementation
COPY predictor.py predictor.py

# Install required dependencies, including optional Feast
RUN pip install --no-cache-dir kserve mlflow pandas torch xgboost \
    && pip install --no-cache-dir feast || true

# Run the predictor with KFServing ModelServer
CMD ["python", "predictor.py"]
