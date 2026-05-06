import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analyze_video import analyze


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fake/real for one video with trained fusion model.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model-json", default="models/fusion_model.json", help="Fusion model json")
    parser.add_argument("--out-json", default="", help="Optional output json path")
    args = parser.parse_args()

    model = json.loads(Path(args.model_json).read_text(encoding="utf-8"))
    result = analyze(args.video)
    scores = result.get("scores", {})

    feats = np.array([float(scores.get(name, 0.0)) for name in model["feature_names"]], dtype=np.float64)
    if bool(model.get("standardize", False)):
        mu = np.array(model.get("scaler_mean") or [], dtype=np.float64)
        sigma = np.array(model.get("scaler_std") or [], dtype=np.float64)
        if len(mu) == len(feats) and len(sigma) == len(feats):
            sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
            feats = (feats - mu) / sigma_safe
    z = float(np.dot(np.array(feats, dtype=np.float64), np.array(model["weights"], dtype=np.float64)) + float(model["bias"]))
    p_fake = _sigmoid(z)
    threshold = float(model.get("threshold", 0.5))

    if abs(p_fake - threshold) <= 0.05:
        label = "UNCERTAIN"
    elif p_fake >= threshold:
        label = "FAKE"
    else:
        label = "REAL"

    out = {
        "video_path": str(Path(args.video).resolve()),
        "model_path": str(Path(args.model_json).resolve()),
        "modal_scores": {k: float(scores.get(k, 0.0)) for k in model["feature_names"]},
        "p_fake": float(p_fake),
        "threshold": threshold,
        "label": label,
        "pipeline_details": result,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

