"""Training utilities integrating XGBoost and Vision Transformer with MLflow."""

from __future__ import annotations

import argparse
from typing import Any


def train_xgboost(learning_rate: float, max_depth: int, n_estimators: int) -> Any:
    """Train an XGBoost model with MLflow autologging."""
    import mlflow
    import mlflow.xgboost  # noqa: F401 - needed for autologging
    import xgboost as xgb
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    mlflow.xgboost.autolog()

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "multi:softprob",
        "num_class": len(set(y_train)),
        "learning_rate": learning_rate,
        "max_depth": max_depth,
    }

    with mlflow.start_run():
        booster = xgb.train(params, dtrain, num_boost_round=n_estimators)
        preds = booster.predict(dtest).argmax(axis=1)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
    return booster


def train_vit(learning_rate: float, epochs: int, batch_size: int) -> Any:
    """Train a Vision Transformer model with MLflow autologging."""
    import mlflow
    import mlflow.pytorch  # noqa: F401 - needed for autologging
    import timm
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import FakeData

    mlflow.pytorch.autolog()

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    dataset = FakeData(
        size=100,
        image_size=(3, 224, 224),
        num_classes=10,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=10
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with mlflow.start_run():
        model.train()
        for _ in range(epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Simple evaluation on the training set
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total if total else 0.0
        mlflow.log_metric("accuracy", acc)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost or Vision Transformer models"
    )
    parser.add_argument("--model", choices=["xgboost", "vit"], required=True)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.model == "xgboost":
        train_xgboost(args.learning_rate, args.max_depth, args.n_estimators)
    else:
        train_vit(args.learning_rate, args.epochs, args.batch_size)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()

