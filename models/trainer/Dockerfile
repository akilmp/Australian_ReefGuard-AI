FROM python:3.10-slim
WORKDIR /app
COPY models/trainer models/trainer
RUN pip install --no-cache-dir mlflow xgboost torch torchvision scikit-learn
CMD ["python", "models/trainer/train.py"]
