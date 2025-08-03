# KFP v2 Pipelines

Kubeflow Pipelines v2 definitions for ReefGuard AI.

## Training Pipeline

The training pipeline submits a Katib experiment that tunes the XGBoost
hyperparameters passed to `models/trainer/train.py` via
`--xgb-learning-rate` and `--xgb-max-depth`.

Recompile the pipeline JSON after making changes:

```bash
python - <<'PY'
from kfp import compiler
from pipelines.kfp_v2.training_pipeline import training_pipeline
compiler.Compiler().compile(training_pipeline, package_path='pipelines/kfp_v2/training_pipeline.json')
PY
```
