from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.visual_model import get_model


def _softmax_fake_prob(logits: torch.Tensor) -> float:
    if logits.shape[-1] == 1:
        return float(torch.sigmoid(logits)[0].item())
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, 1].item())


def _heuristic_frame_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(np.var(lap))
    # Aşırı düşük detay + aşırı yüksek detay (ringing/oversharpen) birlikte şüphe üretir.
    low_detail = np.clip((45.0 - lap_var) / 45.0, 0.0, 1.0)
    high_ringing = np.clip((lap_var - 420.0) / 650.0, 0.0, 1.0)
    return float(np.clip(0.62 * low_detail + 0.38 * high_ringing, 0.0, 1.0))


def _normalize_args(topk, save_gradcam):
    # Eski çağrılarla uyumluluk: visual_score(dir, True) -> save_gradcam=True
    if isinstance(topk, bool):
        return 3, bool(topk)
    return int(topk), bool(save_gradcam)


def visual_score(frames_dir, topk=3, save_gradcam=True, out_dir="data/gradcam"):
    del out_dir  # şu sürümde burada yazma yapılmıyor
    topk, save_gradcam = _normalize_args(topk, save_gradcam)
    del save_gradcam

    model, _target_layer, device = get_model()
    paths = sorted(Path(frames_dir).glob("*.jpg"))
    if not paths:
        return 0.0, []

    scored: List[Tuple[str, float]] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue

        if model is not None:
            img299 = cv2.resize(img, (299, 299))
            rgb = cv2.cvtColor(img299, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model.model(x) if hasattr(model, "model") else model(x)
                prob_fake = _softmax_fake_prob(logits)
        else:
            prob_fake = _heuristic_frame_score(img)

        scored.append((str(path), float(prob_fake)))

    if not scored:
        return 0.0, []

    sv = float(np.mean([score for _, score in scored]))
    top_frames = sorted(scored, key=lambda item: item[1], reverse=True)[: max(1, topk)]
    return sv, top_frames
