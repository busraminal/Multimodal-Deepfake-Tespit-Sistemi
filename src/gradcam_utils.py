# src/gradcam_utils.py
import cv2
import numpy as np
import torch

def _normalize(x):
    x = x - x.min()
    return x / (x.max() + 1e-8)

@torch.no_grad()
def _forward_logits(model, x):
    out = model(x)
    # çıktı (B,2) ise fake index=1 varsayımı
    return out

def gradcam(model, target_layer, img_bgr, device, class_idx=1, input_size=299):
    model.eval()
    base = model.model if hasattr(model, "model") else model

    # preprocess
    img = cv2.resize(img_bgr, (input_size, input_size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device)
    x.requires_grad_(True)

    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = base(x) if base is not model else model(x)
    score = logits[:, class_idx].sum()

    model.zero_grad(set_to_none=True)
    score.backward()

    h1.remove(); h2.remove()

    A = activations[0]          # (1,C,H,W)
    dA = gradients[0]           # (1,C,H,W)

    w = dA.mean(dim=(2,3), keepdim=True)  # (1,C,1,1)
    cam = (w * A).sum(dim=1, keepdim=False)  # (1,H,W)
    cam = torch.relu(cam)[0].detach().cpu().numpy()

    cam = _normalize(cam)
    cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_bgr, 0.55, heat, 0.45, 0)
    return overlay, float(cam.max())
