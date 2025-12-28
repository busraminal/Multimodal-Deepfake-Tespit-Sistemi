import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# src/visual_model.py
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/faceforensics/full/full_c23.p"  # sende bu var

def _pick_target_layer(m):
    # Xception türevleri için güvenli hedef
    for name in ["conv4", "layer4"]:
        if hasattr(m, name):
            return getattr(m, name)
    # olmadıysa son conv bul
    last_conv = None
    for mod in m.modules():
        if mod.__class__.__name__.lower().find("conv") >= 0:
            last_conv = mod
    return last_conv

def load_model():
    # Torch 2.6+ için weights_only=False gerekebiliyor
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # checkpoint bazen direkt nn.Module (TransferModel) geliyor → onu kullan
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(DEVICE).eval()
        base = model.model if hasattr(model, "model") else model
        target_layer = _pick_target_layer(base)
        return model, target_layer

    # dict geldiyse state_dict bul
    state = ckpt.get("state_dict", ckpt)
    raise RuntimeError("Checkpoint dict geldi ama bu projede TransferModel kullanıyoruz. MODEL_PATH yanlış olabilir.")

MODEL, TARGET_LAYER = load_model()

def get_model():
    return MODEL, TARGET_LAYER, DEVICE
