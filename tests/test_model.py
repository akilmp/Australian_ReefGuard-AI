import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import nn

# Ensure repository root on path so ``models`` can be imported when tests are
# executed from a clean environment.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - sanity check
    sys.path.insert(0, str(ROOT))

from models.trainer.model import EnsembleModel, build_model


def test_build_model_returns_ensemble():
    model = build_model()
    assert isinstance(model, EnsembleModel)
    # basic sanity checks on inner models
    assert isinstance(model.vit, nn.Module)
    from xgboost import XGBClassifier

    assert isinstance(model.xgb_model, XGBClassifier)


def test_ensemble_fit_predict_calls_underlying_models():
    class DummyViT(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
            batch = x.shape[0]
            return torch.ones((batch, 4))

    vit = DummyViT()
    xgb = MagicMock()
    xgb.predict.return_value = np.array([0, 1])

    ensemble = EnsembleModel(vit=vit, xgb_model=xgb)

    images = np.zeros((2, 3, 32, 32), dtype=np.float32)
    labels = np.array([0, 1])

    ensemble.fit(images, labels)
    # Ensure XGBoost received features with correct shape
    args, _ = xgb.fit.call_args
    assert args[0].shape == (2, 4)

    preds = ensemble.predict(images)
    xgb.predict.assert_called_once()
    assert preds.shape == (2,)
