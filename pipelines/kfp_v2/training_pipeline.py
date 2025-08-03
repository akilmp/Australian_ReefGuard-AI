"""Kubeflow Pipelines v2 training pipeline invoking Katib Bayesian HPO."""

from __future__ import annotations

import json
from typing import Dict

try:
    from kfp import dsl
except ImportError:  # pragma: no cover - optional dependency
    dsl = None  # type: ignore

    def _identity_decorator(*args, **kwargs):
        def _wrap(func):
            return func

        return _wrap

    dsl = type(
        "dsl",
        (),
        {
            "pipeline": _identity_decorator,
            "container_component": _identity_decorator,
            "ContainerSpec": object,
        },
    )  # type: ignore


@dsl.container_component
def katib_experiment(experiment_spec: str):
    """Launch a Katib experiment using the provided experiment spec."""
    return dsl.ContainerSpec(
        image="docker.io/kubeflowkatib/katib-launcher:latest",
        command=[
            "python",
            "-m",
            "kubeflow.katib.launcher",
            "--experiment-name",
            "reefguard-bayesian",
            "--experiment-namespace",
            "kubeflow",
            "--experiment-spec",
            experiment_spec,
        ],
    )


@dsl.pipeline(
    name="reefguard-training-pipeline",
    description="Training pipeline with Katib Bayesian HPO",
)
def training_pipeline():
    """Construct the pipeline that submits a Katib Bayesian optimization experiment."""
    experiment_spec: Dict = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {"name": "reefguard-bayesian"},
        "spec": {
            "objective": {
                "type": "maximize",
                "goal": 0.9,
                "objectiveMetricName": "ensemble_accuracy",
            },
            "algorithm": {"algorithmName": "bayesianoptimization"},
            "parameters": [
                {
                    "name": "xgbLearningRate",
                    "parameterType": "double",
                    "feasibleSpace": {"min": "0.01", "max": "0.2"},
                },
                {
                    "name": "xgbMaxDepth",
                    "parameterType": "int",
                    "feasibleSpace": {"min": "3", "max": "10"},
                },
                {
                    "name": "xgbNEstimators",
                    "parameterType": "int",
                    "feasibleSpace": {"min": "50", "max": "200"},
                },
                {
                    "name": "vitLr",
                    "parameterType": "double",
                    "feasibleSpace": {"min": "0.0001", "max": "0.01"},
                },
                {
                    "name": "vitEpochs",
                    "parameterType": "int",
                    "feasibleSpace": {"min": "1", "max": "10"},
                },
                {
                    "name": "vitBatchSize",
                    "parameterType": "int",
                    "feasibleSpace": {"min": "8", "max": "64"},
                },
            ],
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {
                        "name": "xgbLearningRate",
                        "description": "Learning rate for XGBoost",
                        "reference": "xgbLearningRate",
                    },
                    {
                        "name": "xgbMaxDepth",
                        "description": "Max depth for XGBoost",
                        "reference": "xgbMaxDepth",
                    },
                    {
                        "name": "xgbNEstimators",
                        "description": "Number of trees for XGBoost",
                        "reference": "xgbNEstimators",
                    },
                    {
                        "name": "vitLr",
                        "description": "Learning rate for Vision Transformer",
                        "reference": "vitLr",
                    },
                    {
                        "name": "vitEpochs",
                        "description": "Training epochs for Vision Transformer",
                        "reference": "vitEpochs",
                    },
                    {
                        "name": "vitBatchSize",
                        "description": "Batch size for Vision Transformer",
                        "reference": "vitBatchSize",
                    },
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": "ttl.sh/reefguard-trainer-1754218355:24h",
                                        "command": [
                                            "python",
                                            "models/trainer/train.py",
                                        ],
                                        "args": [
                                            "--xgb-learning-rate",
                                            "${trialParameters.xgbLearningRate}",
                                            "--xgb-max-depth",
                                            "${trialParameters.xgbMaxDepth}",
                                            "--xgb-n-estimators",
                                            "${trialParameters.xgbNEstimators}",
                                            "--vit-lr",
                                            "${trialParameters.vitLr}",
                                            "--vit-epochs",
                                            "${trialParameters.vitEpochs}",
                                            "--vit-batch-size",
                                            "${trialParameters.vitBatchSize}",
                                        ],
                                    }
                                ],
                                "restartPolicy": "Never",
                            }
                        }
                    },
                },
            },
        },
    }

    # Submit the Katib experiment within the pipeline
    katib_experiment(experiment_spec=json.dumps(experiment_spec))
