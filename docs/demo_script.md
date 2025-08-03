# ReefGuard AI Demo Script

This walkthrough demonstrates an end-to-end run of the ReefGuard AI pipeline on a developer workstation.

1. **Clone and install**
   ```bash
   git clone https://github.com/example/Australian_ReefGuard-AI.git
   cd Australian_ReefGuard-AI
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Launch local services** – start MLflow for experiment tracking and apply Feast feature definitions.
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
   feast apply
   ```
3. **Ingest sample data** – run a lightweight ETL pipeline to pull a small dataset.
   ```bash
   python pipelines/kfp_v2/etl_pipeline.py
   ```
4. **Train a model** – execute the training script on the sample data.
   ```bash
   python models/trainer/train.py
   ```
5. **Serve the model** – launch a simple inference service.
   ```bash
   python models/inference/predictor.py &
   ```
6. **Query the endpoint** – send a test request and review the prediction.
   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"instances": [{"sst_celsius": 28.4, "turbidity_ntu": 3.1}]}' \
        http://localhost:8000/predict
   ```
7. **Monitor** – import `docs/grafana_dash.json` into Grafana to view PSI drift, latency and heat-map panels.
8. **Clean up** – stop background services and remove temporary files.

These steps can be adapted for live infrastructure by swapping the local commands for their Kubernetes or Terraform equivalents.
