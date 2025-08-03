import argparse
import os
from typing import Any, Dict

import mlflow
import mlflow.pytorch
import mlflow.xgboost
import xgboost as xgb
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vit_b_16
from torchvision.transforms import Resize


DEFAULT_MODEL_NAME = "reefguard-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple model and register with MLflow")
    parser.add_argument("--experiment", default="reefguard", help="MLflow experiment name")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Registered model name")
    parser.add_argument("--xgb-max-depth", type=int, default=3, help="XGBoost max tree depth")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1, help="XGBoost learning rate")
    parser.add_argument("--xgb-n-estimators", type=int, default=100, help="Number of trees in XGBoost")
    parser.add_argument("--vit-lr", type=float, default=1e-3, help="Vision Transformer learning rate")
    parser.add_argument("--vit-epochs", type=int, default=5, help="Number of training epochs for ViT")
    parser.add_argument("--vit-batch-size", type=int, default=16, help="Batch size for ViT training")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Log hyperparameters for Katib
        mlflow.log_params(
            {
                "xgb_max_depth": args.xgb_max_depth,
                "xgb_learning_rate": args.xgb_learning_rate,
                "xgb_n_estimators": args.xgb_n_estimators,
                "vit_lr": args.vit_lr,
                "vit_epochs": args.vit_epochs,
                "vit_batch_size": args.vit_batch_size,
            }
        )

        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            n_estimators=args.xgb_n_estimators,
            objective="multi:softprob",
        )
        xgb_model.fit(x_train, y_train)
        xgb_preds = xgb_model.predict(x_test)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        mlflow.log_metric("xgb_accuracy", float(xgb_acc))
        mlflow.xgboost.log_model(xgb_model, artifact_path="xgb-model")
        xgb_probs = xgb_model.predict_proba(x_test)

        # Prepare data for Vision Transformer
        resize = Resize((224, 224))
        x_train_img = torch.tensor(x_train, dtype=torch.float32).view(-1, 1, 2, 2)
        x_test_img = torch.tensor(x_test, dtype=torch.float32).view(-1, 1, 2, 2)
        x_train_img = resize(x_train_img).repeat(1, 3, 1, 1)
        x_test_img = resize(x_test_img).repeat(1, 3, 1, 1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(x_train_img, y_train_tensor),
            batch_size=args.vit_batch_size,
            shuffle=True,
        )

        # Train Vision Transformer
        vit_model = vit_b_16(weights=None)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, len(data.target_names)
        )
        optimizer = torch.optim.Adam(vit_model.parameters(), lr=args.vit_lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(args.vit_epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = vit_model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        vit_model.eval()
        with torch.no_grad():
            vit_logits = vit_model(x_test_img)
            vit_probs = torch.softmax(vit_logits, dim=1)
            vit_preds = vit_probs.argmax(dim=1).numpy()
        vit_acc = accuracy_score(y_test, vit_preds)
        mlflow.log_metric("vit_accuracy", float(vit_acc))
        mlflow.pytorch.log_model(vit_model, artifact_path="vit-model")

        # Ensemble predictions
        ensemble_probs = (xgb_probs + vit_probs.numpy()) / 2.0
        ensemble_preds = ensemble_probs.argmax(axis=1)
        ensemble_acc = accuracy_score(y_test, ensemble_preds)
        mlflow.log_metric("ensemble_accuracy", float(ensemble_acc))

        # Log ensemble model as a PyFunc model and register
        class EnsembleModel(mlflow.pyfunc.PythonModel):
            def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
                import mlflow
                import torch

                self.xgb_model = mlflow.xgboost.load_model(context.artifacts["xgb"])
                self.vit_model = mlflow.pytorch.load_model(context.artifacts["vit"])
                self.vit_model.eval()
                self._torch = torch

            def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Dict[str, Any]):
                x_tab = model_input["tabular"]
                x_img = model_input["image"]
                xgb_probs = self.xgb_model.predict_proba(x_tab)
                with self._torch.no_grad():
                    vit_probs = self._torch.softmax(self.vit_model(x_img), dim=1).numpy()
                return (xgb_probs + vit_probs) / 2.0

        xgb_path = mlflow.artifacts.download_artifacts(artifact_path="xgb-model")
        vit_path = mlflow.artifacts.download_artifacts(artifact_path="vit-model")
        mlflow.pyfunc.log_model(
            artifact_path="ensemble-model",
            python_model=EnsembleModel(),
            artifacts={"xgb": xgb_path, "vit": vit_path},
        )
        model_uri = mlflow.get_artifact_uri("ensemble-model")
        result = mlflow.register_model(model_uri, args.model_name)
        print(f"Registered model {result.name} version {result.version}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    cli_args = parse_args()
    mlflow.set_experiment(cli_args.experiment)
    train(cli_args)

