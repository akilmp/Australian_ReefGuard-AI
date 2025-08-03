import argparse
import os

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DEFAULT_MODEL_NAME = "reefguard-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple model and register with MLflow")
    parser.add_argument("--experiment", default="reefguard", help="MLflow experiment name")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Registered model name")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        clf = LogisticRegression(max_iter=200)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.sklearn.log_model(clf, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")
        result = mlflow.register_model(model_uri, args.model_name)
        print(f"Registered model {result.name} version {result.version}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    cli_args = parse_args()
    mlflow.set_experiment(cli_args.experiment)
    train(cli_args)
