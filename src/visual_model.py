import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATH = "models/faceforensics/full/full_c23.p"

_MODEL = None
_TARGET_LAYER = None
_LOAD_ATTEMPTED = False
_LOAD_ERROR = None


def _pick_target_layer(model):
    for name in ("conv4", "layer4"):
        if hasattr(model, name):
            return getattr(model, name)
    last_conv = None
    for mod in model.modules():
        if "conv" in mod.__class__.__name__.lower():
            last_conv = mod
    return last_conv


def _resolve_model_path() -> str:
    return os.environ.get("DF_VISUAL_MODEL_PATH", DEFAULT_MODEL_PATH)


def _load_model_once() -> None:
    global _MODEL, _TARGET_LAYER, _LOAD_ATTEMPTED, _LOAD_ERROR
    if _LOAD_ATTEMPTED:
        return
    _LOAD_ATTEMPTED = True

    model_path = _resolve_model_path()
    if not Path(model_path).exists():
        _LOAD_ERROR = f"Visual model not found: {model_path}"
        return

    try:
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            _MODEL = ckpt.to(DEVICE).eval()
            base = _MODEL.model if hasattr(_MODEL, "model") else _MODEL
            _TARGET_LAYER = _pick_target_layer(base)
            return
        _LOAD_ERROR = f"Unsupported checkpoint format in {model_path}"
    except Exception as exc:
        _LOAD_ERROR = str(exc)


def get_model() -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module], str]:
    _load_model_once()
    return _MODEL, _TARGET_LAYER, DEVICE


def get_model_status() -> Tuple[bool, str]:
    _load_model_once()
    if _MODEL is not None:
        return True, "ok"
    return False, _LOAD_ERROR or "unknown error"
