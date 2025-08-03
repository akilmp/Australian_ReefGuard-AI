import sys
from pathlib import Path
import types
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - sanity check
    sys.path.insert(0, str(ROOT))

# Provide a minimal stub for torchvision so ``models.trainer.model`` imports
torchvision_stub = types.ModuleType("torchvision")
models_stub = types.ModuleType("models")


def _vit_b_16(weights=None):  # pragma: no cover - simple stub
    model = nn.Identity()
    model.heads = nn.Identity()
    return model


models_stub.vit_b_16 = _vit_b_16
torchvision_stub.models = models_stub
sys.modules.setdefault("torchvision", torchvision_stub)
sys.modules.setdefault("torchvision.models", models_stub)

# Minimal stub for kserve used by Predictor
kserve_stub = types.ModuleType("kserve")


class _KFModel:  # pragma: no cover - simple placeholder
    def __init__(self, name: str):
        self.name = name


class _ModelServer:  # pragma: no cover - placeholder
    def start(self, models):
        return None


kserve_stub.KFModel = _KFModel
kserve_stub.ModelServer = _ModelServer
sys.modules.setdefault("kserve", kserve_stub)
