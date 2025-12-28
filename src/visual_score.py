# src/visual_score.py
print("NEW visual_score.py LOADED <<<")

import cv2
import numpy as np
from pathlib import Path
import torch

from src.visual_model import get_model
from src.gradcam_utils import gradcam

def _softmax_fake_prob(logits):
    if logits.shape[-1] == 1:
        return float(torch.sigmoid(logits)[0].item())
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, 1].item())  # fake=1

def visual_score(frames_dir, topk=3, save_gradcam=True, out_dir="data/gradcam"):
    MODEL, TARGET_LAYER, DEVICE = get_model()
    frames_dir = Path(frames_dir)
    paths = sorted([p for p in frames_dir.glob("*.jpg")])

    if len(paths) == 0:
        return 0.0, []

    scored = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img299 = cv2.resize(img, (299, 299))
        rgb = cv2.cvtColor(img299, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = (MODEL.model(x) if hasattr(MODEL, "model") else MODEL(x))
            prob_fake = _softmax_fake_prob(logits)

        scored.append((str(p), prob_fake, img))

    if not scored:
        return 0.0, []

    # Sv: ortalama fake olasılığı (istersen max da yaparsın)
    Sv = float(np.mean([s[1] for s in scored]))

    top = sorted(scored, key=lambda t: t[1], reverse=True)[:topk]

    top_frames = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for fp, prob, img in top:
        if save_gradcam:
            overlay, cam_strength = gradcam(
                model=MODEL,
                target_layer=TARGET_LAYER,
                img_bgr=img,
                device=DEVICE,
                class_idx=1
            )
            out_path = str(Path(out_dir) / (Path(fp).stem + "_cam.jpg"))
            cv2.imwrite(out_path, overlay)
            top_frames.append((fp, prob, out_path, cam_strength))
        else:
            top_frames.append((fp, prob, None, None))

    return Sv, top_frames
