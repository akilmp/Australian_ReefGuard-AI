# Simple Dockerfile for KFServing Predictor
FROM python:3.9-slim

WORKDIR /app

# Copy predictor implementation
COPY models/inference/predictor.py predictor.py

# Install required dependencies
RUN pip install --no-cache-dir kserve

# Run the predictor with KFServing ModelServer
CMD ["python", "predictor.py"]
