# MLflow Helm Deployment

This chart values file configures an MLflow tracking server with a SQLite backend and S3 artifact store. Deploy using the official MLflow chart:

```bash
helm repo add mlflow https://mlflow.github.io/mlflow-helm
helm upgrade --install mlflow mlflow/mlflow -f values.yaml
```

Set `MLFLOW_TRACKING_URI` to the service URL for the training code.
