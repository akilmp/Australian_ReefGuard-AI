"""Model definition for training.

This module provides a small ensemble that marries a Vision Transformer
feature extractor with an XGBoost classifier.  The Vision Transformer is
responsible for converting images into dense feature vectors.  These
features are then used by the XGBoost model to produce final predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torchvision.models as tv_models
import xgboost as xgb


@dataclass
class EnsembleModel:
    """Ensemble of a Vision Transformer and an XGBoost classifier.

    Parameters
    ----------
    vit:
        The Vision Transformer feature extractor.  The classifier head is
        expected to be removed so that calling the model returns feature
        embeddings.
    xgb_model:
        The XGBoost classifier receiving the Vision Transformer embeddings.
    """

    vit: nn.Module
    xgb_model: xgb.XGBClassifier

    def _extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract embeddings from images using the Vision Transformer.

        Parameters
        ----------
        images:
            Array of shape ``(N, C, H, W)`` representing a batch of images.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(N, F)`` where ``F`` is the embedding
            dimension of the Vision Transformer.
        """

        self.vit.eval()
        with torch.no_grad():
            if isinstance(images, np.ndarray):
                # ``torch.from_numpy`` cannot be used in some environments where
                # PyTorch was built against an older NumPy version.  Converting
                # via ``tolist`` avoids reliance on NumPy's C-API which keeps
                # the function lightweight for small test tensors.
                tensor = torch.tensor(images.tolist(), dtype=torch.float32)
            else:  # assume ``torch.Tensor``
                tensor = images.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            features = self.vit(tensor)
        # Avoid ``Tensor.numpy`` which requires a NumPy version matching the
        # one used during PyTorch compilation.  Converting via ``tolist`` keeps
        # the dependency minimal and works in constrained environments.
        return np.array(features.detach().cpu().tolist())

    def fit(self, images: np.ndarray, labels: np.ndarray, **kwargs: Any) -> None:
        """Train the ensemble on the provided images and labels.

        Parameters
        ----------
        images:
            Training images as a numpy array.
        labels:
            Corresponding labels for ``images``.
        **kwargs:
            Additional keyword arguments passed to
            :meth:`xgboost.XGBClassifier.fit`.
        """

        features = self._extract_features(images)
        self.xgb_model.fit(features, labels, **kwargs)

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict labels for the given images.

        Parameters
        ----------
        images:
            Images to classify.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """

        features = self._extract_features(images)
        return self.xgb_model.predict(features)


def build_model() -> EnsembleModel:
    """Create a default Vision Transformer + XGBoost ensemble.

    The Vision Transformer is initialised without pretrained weights and its
    classification head is replaced by an identity layer so that the model
    outputs raw embeddings.

    Returns
    -------
    EnsembleModel
        A ready-to-train ensemble instance.
    """

    vit = tv_models.vit_b_16(weights=None)
    vit.heads = nn.Identity()
    xgb_model = xgb.XGBClassifier()
    return EnsembleModel(vit=vit, xgb_model=xgb_model)
