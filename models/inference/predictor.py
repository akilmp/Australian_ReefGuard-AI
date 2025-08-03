"""Predictor implementation for KFServing."""

from typing import Dict, List

import kserve


class Predictor(kserve.KFModel):
    """Simple example predictor for KFServing.

    The model echoes the input instances as predictions. In a real scenario,
    you would load a trained model in :meth:`load` and generate predictions in
    :meth:`predict`.
    """

    def __init__(self, name: str, model_dir: str | None = None):
        super().__init__(name)
        self.model_dir = model_dir
        self.ready = False

    def load(self) -> None:
        """Load model artifacts and mark the model as ready."""
        # Placeholder for model loading logic.
        self.ready = True

    def predict(self, request: Dict) -> Dict[str, List]:
        """Generate predictions from the request payload.

        Parameters
        ----------
        request:
            The request payload passed by KFServing. This implementation
            expects a JSON object with an ``instances`` field containing a
            list of inputs.

        Returns
        -------
        Dict[str, List]
            A dictionary with a ``predictions`` key echoing the provided
            ``instances``.
        """
        instances = request.get("instances", [])
        return {"predictions": instances}


if __name__ == "__main__":
    model = Predictor("reefguard-model")
    model.load()
    kserve.ModelServer().start([model])
