# Model Inference

`predictor.py` exposes a KServe-compatible server that loads an MLflow model and serves predictions.

For local testing, save a model to a directory (e.g., `models/artifacts/latest`) and run:

```bash
python predictor.py --model-dir models/artifacts/latest &
curl -X POST -H "Content-Type: application/json" \
  -d '{"instances": [{"sst": 28.4, "turbidity": 3.1}]}' \
  http://localhost:8080/v1/models/reefguard-model:predict
```
