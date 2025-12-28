# src/fusion.py
import numpy as np


def fuse_scores(
    Sv: float,  # visual fake prob
    Sl: float,  # lip-sync mismatch
    Sb: float,  # blink anomaly
    Sh: float,  # head-pose anomaly
    Sa: float,  # audio artefact
    weights=(0.4, 0.2, 0.15, 0.1, 0.15)
):
    """
    Dönüş:
        Sf (float): [0,1] nihai deepfake skoru
        verdict (str): REAL | SUSPICIOUS | DEEPFAKE | AI_GENERATED / COMPRESSION
    """

    # --- clamp skorlar ---
    Sv, Sl, Sb, Sh, Sa = [float(np.clip(x, 0.0, 1.0)) for x in (Sv, Sl, Sb, Sh, Sa)]

    # --- ağırlıklar ---
    w = np.array(weights, dtype=float)
    w = np.clip(w, 0.0, 1.0)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    wv, wl, wb, wh, wa = w.tolist()

    # --- fusion ---
    Sf = (
        wv * Sv +
        wl * Sl +
        wb * Sb +
        wh * Sh +
        wa * Sa
    )

    # ===== KARAR BLOĞU =====
    if Sv > 0.60 and Sl < 0.30 and Sa < 0.30:
        verdict = "AI_GENERATED / COMPRESSION"
    elif Sf > 0.60:
        verdict = "DEEPFAKE"
    elif Sf < 0.45:
        verdict = "REAL"
    else:
        verdict = "SUSPICIOUS"

    return float(Sf), verdict

# =====================================================
# SCORE INTERPRETATION (VERDICT HELPER)
# =====================================================

def interpret_score(Sf, has_speech=False):
    if not has_speech and Sf < 0.6:
        return (
            "REAL",
            "Sessiz video. Görsel artefaktlar sıkıştırma / ışık kaynaklı."
        )

    if Sf >= 0.75:
        return (
            "DEEPFAKE",
            "Yüksek tutarsızlık: yüz-doku, zamansal kopukluk."
        )

    if 0.55 <= Sf < 0.75:
        return (
            "AI_GENERATED / COMPRESSION",
            "AI üretimi veya ağır sıkıştırma artefaktı."
        )

    return (
        "REAL",
        "Tutarlı yüz, düşük yapaylık izi."
    )
