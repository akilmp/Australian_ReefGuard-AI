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
                "objectiveMetricName": "accuracy",
            },
            "algorithm": {"algorithmName": "bayesianoptimization"},
            "parameters": [
                {
                    "name": "learningRate",
                    "parameterType": "double",
                    "feasibleSpace": {"min": "0.01", "max": "0.2"},
                },
                {
                    "name": "maxDepth",
                    "parameterType": "int",
                    "feasibleSpace": {"min": "3", "max": "10"},
                },
            ],
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {
                        "name": "learningRate",
                        "description": "Learning rate for XGBoost",
                        "reference": "learningRate",
                    },
                    {
                        "name": "maxDepth",
                        "description": "Max depth for XGBoost",
                        "reference": "maxDepth",
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
                                        "image": "python:3.10",
                                        "command": [
                                            "python",
                                            "models/trainer/train.py",
                                        ],
                                        "args": [
                                            "--model",
                                            "xgboost",
                                            "--learning-rate",
                                            "${trialParameters.learningRate}",
                                            "--max-depth",
                                            "${trialParameters.maxDepth}",
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

