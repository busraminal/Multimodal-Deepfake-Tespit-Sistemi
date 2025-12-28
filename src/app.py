# ============================== app.py (1/2) ==============================
# ✅ Tek dosya Streamlit app (Bölüm 1/2)
# ✅ Heuristic heatmap default ON
# ✅ MediaPipe landmark (opsiyonel)
# ✅ Real Grad-CAM (model+torch varsa) toggle
# ✅ Hiçbiri kurulu olmasa bile app çalışır
# ==========================================================================

from __future__ import annotations

import os
import sys
import io
import re
import shutil
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.llm_client import send_to_llm
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import cv2

# Explainability default
top_frames = []        # [(img_path, score, time_sec), ...]
pdf_frames = []        # PDF için güvenli varsayılan

# =========================================================
# 🔧 ZORUNLU PATCH – Streamlit crash fix
# =========================================================
from typing import Dict, List, Tuple

def _get_status_placeholder():
    if "status" not in st.session_state:
        st.session_state.status = st.empty()
    return st.session_state.status

def _get_progress_bar():
    if "progress" not in st.session_state:
        st.session_state.progress = st.progress(0)
    return st.session_state.progress

# combine_cam_and_landmarks İMZA DÜZELTME
def combine_cam_and_landmarks(
    img_rgb: np.ndarray,
    prefer_real_cam: bool,
    alpha_cam: float,
    show_landmarks: bool = True,
):
    cam_kind = "Heuristic"
    cam_img = None

    if prefer_real_cam:
        cam_img = real_gradcam_overlay(img_rgb, alpha=alpha_cam)
        if cam_img is not None:
            cam_kind = "Real Grad-CAM"

    if cam_img is None:
        cam_img = heuristic_heatmap_overlay(img_rgb, alpha=alpha_cam)
        cam_kind = "Heuristic"

    if show_landmarks:
        cam_img = draw_face_landmarks(cam_img)

    return cam_img, cam_kind


# =========================================================
# (OPSİYONEL) PDF
# =========================================================
HAS_REPORTLAB = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader
except Exception:
    HAS_REPORTLAB = False
    A4 = None
    rl_canvas = None
    ImageReader = None

# =========================================================
# (OPSİYONEL) MediaPipe FaceMesh
# =========================================================
HAS_MP = False
mp = None
try:
    import mediapipe as mp  # type: ignore
    HAS_MP = True
except Exception:
    HAS_MP = False
    mp = None

# =========================================================
# (OPSİYONEL) Grad-CAM (pytorch-grad-cam)
# =========================================================
HAS_TORCH_CAM = False
torch = None
GradCAM = None
show_cam_on_image = None
try:
    import torch  # type: ignore
    from pytorch_grad_cam import GradCAM  # type: ignore
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore
    HAS_TORCH_CAM = True
except Exception:
    HAS_TORCH_CAM = False
    torch = None
    GradCAM = None
    show_cam_on_image = None

# =========================================================
# PATH / LOG
# =========================================================
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(THIS_DIR.parent))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import absl.logging  # type: ignore
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

# FFmpeg (Windows) — senin path’in
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ.get("PATH", "")

# =========================================================
# BACKEND IMPORTLAR (zorunlu)
# =========================================================
# Not: Bunlar projende yoksa zaten app mantıken çalışamaz.
from src.media_io import extract_audio, extract_frames
from src.asr_text import transcribe
from src.lip_sync import lip_mismatch_score
from src.visual_score import visual_score
from src.fusion import interpret_score
from src.biomech import blink_score, headpose_score
from src.audio_artefact import audio_gan_score

# =========================================================
# (OPSİYONEL) Model erişimi (Real Grad-CAM için)
# - src/visual_model.py içinde `model` ve `target_layer` export edersen çalışır.
# =========================================================
visual_model = None
visual_target_layer = None
try:
    from src.visual_model import model as visual_model  # type: ignore
    from src.visual_model import target_layer as visual_target_layer  # type: ignore
except Exception:
    visual_model = None
    visual_target_layer = None

# =========================================================
# SABİTLER
# =========================================================
VIDEO_TMP = "data/tmp_upload.mp4"
AUDIO_PATH = "data/audio/gui.wav"
FRAMES_DIR = "data/frames/gui_mouth"

MAX_VIDEO_WIDTH = 720
TEXT_MIN_CHARS_FOR_LIPSYNC = 12

# =========================================================
# UI STATE (Streamlit rerun-safe)
# =========================================================
if "status_placeholder" not in st.session_state:
    st.session_state.status_placeholder = None
if "progress_bar" not in st.session_state:
    st.session_state.progress_bar = None


def _get_status_placeholder():
    if st.session_state.status_placeholder is None:
        st.session_state.status_placeholder = st.empty()
    return st.session_state.status_placeholder


def _get_progress_bar():
    if st.session_state.progress_bar is None:
        st.session_state.progress_bar = st.progress(0)
    return st.session_state.progress_bar


# =========================================================
# YARDIMCI
# =========================================================
def _ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/audio", exist_ok=True)
    os.makedirs("data/frames", exist_ok=True)


def _reset_run_dirs():
    _ensure_dirs()
    if os.path.isdir(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    if os.path.isfile(AUDIO_PATH):
        try:
            os.remove(AUDIO_PATH)
        except Exception:
            pass


def _render_video(path: str):
    # Streamlit video bileşeni local file’ı direkt açar ama bazen path issue çıkar.
    # HTML video ile daha stabil.
    st.markdown(
        f"""
        <div style="max-width:{MAX_VIDEO_WIDTH}px;margin:0 auto;">
            <video controls style="width:100%;border-radius:14px;">
                <source src="{path}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(d)
    except Exception:
        return float(d)


def _clip01(x: Any) -> float:
    return float(np.clip(_safe(x, 0.0), 0.0, 1.0))


def _fusion_v3(Sv: float, Sl: float, Sb: float, Sh: float, Sa: float, has_speech: bool) -> Tuple[float, Dict[str, float]]:
    # Konuşmalı vs sessiz mod
    if has_speech:
        w = {"v": 0.45, "l": 0.20, "b": 0.10, "h": 0.10, "a": 0.15}
    else:
        w = {"v": 0.55, "l": 0.00, "b": 0.15, "h": 0.10, "a": 0.20}

    Sf = (
        w["v"] * Sv
        + w["l"] * Sl
        + w["b"] * Sb
        + w["h"] * Sh
        + w["a"] * Sa
    )
    return float(np.clip(Sf, 0.0, 1.0)), w


def _score_color(sf: float) -> str:
    return "#bb00c8" if sf < 0.35 else "#ff005d" if sf < 0.7 else "#0079d5"


def _badge(has_speech: bool) -> str:
    if has_speech:
        return "<span style='background:#123b2b;color:#b9ffdf;padding:6px 10px;border-radius:999px;font-weight:800;'>🟢 Konuşmalı Video</span>"
    return "<span style='background:#3b2f12;color:#ffe7b9;padding:6px 10px;border-radius:999px;font-weight:800;'>🟡 Sessiz / Az Konuşmalı Video</span>"


def _compact_bar_row(scores: Dict[str, float]):
    def one(label: str, val: float) -> str:
        pct = int(_clip01(val) * 100)
        return f"""
        <div style="flex:1;min-width:120px">
          <div style="display:flex;justify-content:space-between;gap:8px;font-size:12px;color:#ddd;margin-bottom:4px">
            <span><b>{label}</b></span><span>{val:.2f}</span>
          </div>
          <div style="background:#1e1e1e;border-radius:999px;height:8px;overflow:hidden">
            <div style="width:{pct}%;height:8px;border-radius:999px;background:linear-gradient(90deg,#00c6ff,#0072ff)"></div>
          </div>
        </div>
        """

    items = "".join([one(k, float(v)) for k, v in scores.items()])
    st.markdown(
        f"""
        <div style="display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end">
          {items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _parse_frame_idx(img_path: str, fallback: int) -> int:
    name = Path(img_path).stem
    m = re.findall(r"(\d+)", name)
    if m:
        try:
            return int(m[-1])
        except Exception:
            return fallback
    return fallback


def _get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    if fps and fps > 1e-3:
        return float(fps)
    return 25.0


def frame_to_timecode(frame_idx: int, fps: float) -> str:
    sec = frame_idx / max(fps, 1e-6)
    mm = int(sec // 60)
    ss = sec - 60 * mm
    return f"{mm:02d}:{ss:05.2f}"


# =========================================================
# Plotly Gauge
# =========================================================
def render_sf_gauge(sf: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(sf),
            number={"font": {"size": 28}},
            title={"text": "Sf · Nihai Risk Skoru"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": _score_color(sf)},
                "steps": [
                    {"range": [0, 0.35], "color": "#2a002f"},
                    {"range": [0.35, 0.7], "color": "#330014"},
                    {"range": [0.7, 1.0], "color": "#001c33"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)


def frame_explain(score: float) -> str:
    s = _clip01(score)
    if s >= 0.85:
        return "⚠️ Yüksek risk: doku/sınır artefaktı, yapay keskinlik veya yüz çevresi tutarsız"
    if s >= 0.65:
        return "❗ Orta risk: ışık/gölge, mimik akışı veya detay tutarsızlığı"
    if s >= 0.45:
        return "ℹ️ Düşük-orta: küçük tutarsızlıklar var, tek başına kanıt değil"
    return "✅ Doğal: belirgin görsel tutarsızlık yok"


def render_sf_timeline(frame_scores: List[Tuple[str, float]], fps: float, title: str = "Sf (p(fake)) · Zaman Boyunca"):
    if not frame_scores:
        return

    tmp: List[Tuple[int, float]] = []
    for j, item in enumerate(frame_scores):
        p = item[0]
        s = item[1]
        idx = _parse_frame_idx(p, j)
        tmp.append((idx, _clip01(s)))
    tmp.sort(key=lambda x: x[0])

    xs = [idx / max(fps, 1e-6) for idx, _ in tmp]
    ys = [s for _, s in tmp]

    win = 7
    ma: List[float] = []
    for i in range(len(ys)):
        a = max(0, i - win // 2)
        b = min(len(ys), i + win // 2 + 1)
        ma.append(sum(ys[a:b]) / max(1, (b - a)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Sf (frame)"))
    fig.add_trace(go.Scatter(x=xs, y=ma, mode="lines", name=f"Moving Avg (win={win})"))
    fig.update_layout(
        title=title,
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Zaman (sn)",
        yaxis_title="p(fake)",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Explainability: Heuristic Heatmap (default)
# =========================================================
def heuristic_heatmap_overlay(img_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    # Laplacian magnitude heatmap
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mag = np.abs(lap)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(img_rgb, 1 - float(alpha), heat, float(alpha), 0)
    return out


# =========================================================
# Explainability: Real Grad-CAM (model+torch varsa)
# =========================================================
def _torch_ready() -> bool:
    return bool(HAS_TORCH_CAM and (torch is not None) and (GradCAM is not None) and (show_cam_on_image is not None))


def real_gradcam_overlay(img_rgb: np.ndarray, alpha: float = 0.55) -> Optional[np.ndarray]:
    """
    Model/torch/cam yoksa None döner.
    Not: alpha paramı burada sadece API uyumu için; show_cam_on_image zaten overlay yapıyor.
    """
    if not _torch_ready():
        return None
    if visual_model is None or visual_target_layer is None:
        return None

    img_norm = img_rgb.astype(np.float32) / 255.0  # [0,1] RGB
    x = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0)

    # güvenli device
    try:
        visual_model.eval()
    except Exception:
        pass

    try:
        cam = GradCAM(model=visual_model, target_layers=[visual_target_layer])
        grayscale_cam = cam(input_tensor=x)[0]  # HxW
        cam_img = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)
        return cam_img
    except Exception:
        return None


# =========================================================
# Face landmarks (MediaPipe)
# =========================================================
_FACE_MESH = None
_MP_DRAW = None
_MP_STYLE = None

if HAS_MP and mp is not None:
    try:
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        _MP_DRAW = mp.solutions.drawing_utils
        _MP_STYLE = mp.solutions.drawing_styles
    except Exception:
        _FACE_MESH = None
        _MP_DRAW = None
        _MP_STYLE = None


def draw_face_landmarks(img_rgb: np.ndarray) -> np.ndarray:
    if not (HAS_MP and _FACE_MESH is not None and _MP_DRAW is not None and _MP_STYLE is not None and mp is not None):
        return img_rgb

    res = _FACE_MESH.process(img_rgb)
    if not res.multi_face_landmarks:
        return img_rgb

    out = img_rgb.copy()
    for lm in res.multi_face_landmarks:
        _MP_DRAW.draw_landmarks(
            image=out,
            landmark_list=lm,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_MP_STYLE.get_default_face_mesh_contours_style(),
        )
        _MP_DRAW.draw_landmarks(
            image=out,
            landmark_list=lm,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=_MP_STYLE.get_default_face_mesh_iris_connections_style(),
        )
    return out


def combine_cam_and_landmarks(
    img_rgb: np.ndarray,
    prefer_real_cam: bool,
    alpha_cam: float,
    show_landmarks: bool,
) -> Tuple[np.ndarray, str]:
    """
    1) prefer_real_cam True ve real gradcam mümkünse -> Real Grad-CAM
    2) değilse -> Heuristic
    3) show_landmarks True ise üstüne landmark bas
    """
    cam_kind = "Heuristic"
    cam_img: Optional[np.ndarray] = None

    if prefer_real_cam:
        cam_img = real_gradcam_overlay(img_rgb, alpha=float(alpha_cam))
        if cam_img is not None:
            cam_kind = "Real Grad-CAM"

    if cam_img is None:
        cam_img = heuristic_heatmap_overlay(img_rgb, alpha=float(alpha_cam))
        cam_kind = "Heuristic"

    if show_landmarks:
        cam_img = draw_face_landmarks(cam_img)

    return cam_img, cam_kind


# =========================================================
# UI: Method schema
# =========================================================
def render_method_schema():
    st.markdown(
        """
        <div style="border:1px solid rgba(255,255,255,0.15);border-radius:14px;padding:14px">
          <div style="font-weight:900;margin-bottom:8px">🧩 Metod Şeması (1 sayfa)</div>
          <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:center">
            <div style="padding:10px 14px;border-radius:12px;background:#0a1b2a;border:1px solid rgba(255,255,255,0.12)"><b>Video</b> → Kare + Ses</div>
            <div style="font-size:18px">➜</div>
            <div style="padding:10px 14px;border-radius:12px;background:#14122a;border:1px solid rgba(255,255,255,0.12)"><b>Sv</b> Görsel</div>
            <div style="padding:10px 14px;border-radius:12px;background:#14122a;border:1px solid rgba(255,255,255,0.12)"><b>Sl</b> Lip-sync</div>
            <div style="padding:10px 14px;border-radius:12px;background:#14122a;border:1px solid rgba(255,255,255,0.12)"><b>Sb/Sh</b> Biyomek.</div>
            <div style="padding:10px 14px;border-radius:12px;background:#14122a;border:1px solid rgba(255,255,255,0.12)"><b>Sa</b> Audio</div>
            <div style="font-size:18px">➜</div>
            <div style="padding:10px 14px;border-radius:12px;background:#0b2a16;border:1px solid rgba(255,255,255,0.12)"><b>Fusion</b> (moda göre ağırlık)</div>
            <div style="font-size:18px">➜</div>
            <div style="padding:10px 14px;border-radius:12px;background:#2a0b22;border:1px solid rgba(255,255,255,0.12)"><b>Sf</b> Nihai Risk</div>
            <div style="font-size:18px">➜</div>
            <div style="padding:10px 14px;border-radius:12px;background:#0a1b2a;border:1px solid rgba(255,255,255,0.12)"><b>Explain</b>: Timeline + CAM+Landmark + Top Frames</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PDF report
# =========================================================
def generate_pdf_report(
    video_path: str,
    has_speech: bool,
    text: str,
    scores: Dict[str, float],
    weights: Dict[str, float],
    fps: float,
    top_frames: List[Tuple[str, float, str, str, np.ndarray]],  # (img_path, score, timecode, cam_kind, cam_img_rgb)
) -> Optional[bytes]:
    if not HAS_REPORTLAB or rl_canvas is None or A4 is None or ImageReader is None:
        return None

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    W, H = A4  # noqa: F841

    def draw_title(y: float) -> float:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, "Multimodal Deepfake Tespit Raporu")
        c.setFont("Helvetica", 10)
        c.drawString(40, y - 16, f"Tarih: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(40, y - 30, f"Mod: {'Konuşmalı' if has_speech else 'Sessiz/Az konuşma'}   |   FPS: {fps:.2f}")
        return y - 50

    y = H - 50
    y = draw_title(y)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Skor Özeti")
    y -= 18
    c.setFont("Helvetica", 10)
    line = (
        f"Sv={scores.get('Sv',0):.3f}  Sl={scores.get('Sl',0):.3f}  "
        f"Sb={scores.get('Sb',0):.3f}  Sh={scores.get('Sh',0):.3f}  "
        f"Sa={scores.get('Sa',0):.3f}  Sf={scores.get('Sf',0):.3f}"
    )
    c.drawString(40, y, line)
    y -= 16
    wline = (
        f"Ağırlıklar: v={weights.get('v',0):.2f}  l={weights.get('l',0):.2f}  "
        f"b={weights.get('b',0):.2f}  h={weights.get('h',0):.2f}  a={weights.get('a',0):.2f}"
    )
    c.drawString(40, y, wline)
    y -= 26

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Metod (1 sayfa özet)")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(40, y, "Video → Kare+Ses → (Sv,Sl,Sb,Sh,Sa) → Fusion → Sf → Explain (timeline + CAM+landmark)")
    y -= 26

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Transcript (kısa)")
    y -= 18
    c.setFont("Helvetica", 9)
    t = (text or "(yok)").strip().replace("\n", " ")
    t = t[:380] + ("..." if len(t) > 380 else "")
    c.drawString(40, y, t)
    y -= 26

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "En Şüpheli Kareler (CAM + Landmark)")
    y -= 14

    img_w = 240
    img_h = 160
    x0s = [40, 310]
    row_y = y - img_h

    placed = 0
    for (img_path, sc, tc, cam_kind, cam_img_rgb) in top_frames:
        col = placed % 2
        row = placed // 2
        x = x0s[col]
        yy = row_y - row * (img_h + 70)

        if yy < 80:
            c.showPage()
            y2 = H - 50
            y2 = draw_title(y2)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y2, "En Şüpheli Kareler (devam)")
            row_y = y2 - 14 - img_h
            yy = row_y
            placed = 0
            col = 0
            row = 0
            x = x0s[col]

        png_buf = io.BytesIO()
        bgr = cv2.cvtColor(cam_img_rgb, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".png", bgr)
        if ok:
            png_buf.write(enc.tobytes())
            png_buf.seek(0)
            c.drawImage(ImageReader(png_buf), x, yy, width=img_w, height=img_h, preserveAspectRatio=True, anchor="c")

        c.setFont("Helvetica", 9)
        c.drawString(x, yy - 14, f"p(fake)={float(sc):.2f}   time={tc}   CAM={cam_kind}")
        c.drawString(x, yy - 28, frame_explain(float(sc)))

        placed += 1
        if placed >= 4:
            break

    c.save()
    buf.seek(0)
    return buf.read()


# ============================== app.py (2/2) ==============================
# UI + PIPELINE + EXPLAINABILITY + PDF
# ==========================================================================

# =========================================================
# SAYFA
# =========================================================
st.set_page_config(
    page_title="Multimodal Deepfake",
    layout="wide",
    page_icon="🎭"
)

st.markdown(
    """
    <div style="padding:12px;background:#0a1b2a;border-radius:12px;margin-bottom:14px;">
      <h1 style="color:white;text-align:center;margin:0;">🎭 Multimodal Deepfake Tespit Sistemi</h1>
      <p style="color:#cfe8ff;text-align:center;margin:6px 0 0 0;">
        Görsel + Lip-Sync + Biyomekanik + Audio → Risk Skoru + Explainability
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_method_schema()

st.markdown(
    """
    <div style="padding:12px;border:1px solid rgba(255,255,255,0.12);border-radius:12px;margin:12px 0;">
      <div style="font-weight:900;margin-bottom:6px;">ℹ️ Skor Açıklamaları</div>
      <div style="color:#d8d8d8;line-height:1.6;">
        <b>Sv</b>: Görsel tutarsızlık ·
        <b>Sl</b>: Ses–dudak uyumsuzluğu ·
        <b>Sb</b>: Blink anomali ·
        <b>Sh</b>: Head-pose anomali ·
        <b>Sa</b>: Audio artefact ·
        <b>Sf</b>: Ağırlıklı final skor
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# UPLOAD
# =========================================================
file = st.file_uploader(
    "Bir video yükleyin (MP4)",
    type=["mp4"],
    key="video_uploader_main"
)


# Global skorlar (hover paneller için)
Sv = Sl = Sb = Sh = Sa = Sf = 0.0
has_speech = False
w: Dict[str, float] = {"v": 0, "l": 0, "b": 0, "h": 0, "a": 0}
frame_scores: List[Tuple[str, float]] = []
text = ""
fps = 25.0

# =========================================================
# PIPELINE
# =========================================================
if file:
    _reset_run_dirs()

    with open(VIDEO_TMP, "wb") as f:
        f.write(file.getbuffer())

    fps = _get_video_fps(VIDEO_TMP)

    status = _get_status_placeholder()
    progress = _get_progress_bar()

    # ---------------- Sidebar controls ----------------
    st.sidebar.markdown("### 🎛️ Explainability Ayarları")
    prefer_real_cam = st.sidebar.toggle("Gerçek Grad-CAM (model varsa)", value=True)
    show_landmarks = st.sidebar.toggle("Yüz landmark overlay", value=True)
    alpha_cam = st.sidebar.slider("CAM / Heatmap yoğunluğu", 0.20, 0.75, 0.45, 0.05)
    show_timeline = st.sidebar.toggle("Sf zaman grafiği", value=True)

    # ---------------- Layout ----------------
    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.subheader("🎬 Video")
        _render_video(VIDEO_TMP)
        st.caption(f"FPS: {fps:.2f}")

    # ---------------- Pipeline steps ----------------
    status.write("1/7 Ses çıkarılıyor")
    extract_audio(VIDEO_TMP, AUDIO_PATH)
    progress.progress(14)

    status.write("2/7 Kareler çıkarılıyor")
    extract_frames(VIDEO_TMP, FRAMES_DIR)
    progress.progress(28)

    status.write("3/7 Transkript")
    try:
        text = transcribe(AUDIO_PATH, "tr").strip()
    except Exception:
        text = ""
    progress.progress(42)

    status.write("4/7 Lip-sync")
    has_speech = len(text) >= TEXT_MIN_CHARS_FOR_LIPSYNC
    Sl = lip_mismatch_score(AUDIO_PATH, FRAMES_DIR) if has_speech else 0.0
    progress.progress(56)

    status.write("5/7 Görsel skor")
    Sv, frame_scores = visual_score(FRAMES_DIR, True)
    progress.progress(72)

    status.write("6/7 Biyomekanik")
    Sb = blink_score(FRAMES_DIR)
    Sh = headpose_score(FRAMES_DIR)
    progress.progress(86)

    status.write("7/7 Audio")
    Sa, _ = audio_gan_score(AUDIO_PATH, True) if os.path.exists(AUDIO_PATH) else (0.0, {})
    progress.progress(100)

    # ---------------- Fusion ----------------
    Sv = _clip01(Sv)
    Sl = _clip01(Sl)
    Sb = _clip01(Sb)
    Sh = _clip01(Sh)
    Sa = _clip01(Sa)

    Sf, w = _fusion_v3(Sv, Sl, Sb, Sh, Sa, has_speech)

# =========================================================
# UPLOAD
# =========================================================


if file:
    # 🔴 right BURADA OLUŞUR
    left, right = st.columns([1.25, 1.0], gap="large")

    # ================= LLM REASONING =================
    llm_out = None
    try:
        llm_out = send_to_llm(
            video_id="uploaded_video",
            features=[Sv, Sl, Sb, Sh, Sa]
        )
    except Exception:
        llm_out = {
            "confidence": Sf,
            "explanation": "LLM bağlantısı yok, lokal skor kullanıldı."
        }

    # ================= LEFT PANEL =================
    with left:
        st.subheader("🎬 Video")
        _render_video(VIDEO_TMP)

    # ================= RIGHT PANEL =================
    with right:
        st.subheader("📌 Sonuç")
        st.markdown(_badge(has_speech), unsafe_allow_html=True)
        render_sf_gauge(Sf)

        verdict, verdict_msg = interpret_score(Sf)
        st.success(f"Sonuç: **{verdict.upper()}**")
        st.caption(verdict_msg)

        # 🧠 LLM PANELİ
        if llm_out is not None:
            st.markdown("### 🧠 LLM Açıklaması")
            st.metric(
                "Deepfake Confidence (LLM)",
                f"%{int(llm_out.get('confidence', Sf) * 100)}"
            )
            st.write(llm_out.get("explanation", ""))

        with st.expander("📄 Transcript (Whisper)"):
            st.write(text if text else "(Transkript yok)")



    # ---------------- Skor barları ----------------
    st.markdown("---")
    st.subheader("📊 Skorlar (Compact)")
    _compact_bar_row({"Sv": Sv, "Sl": Sl, "Sb": Sb, "Sh": Sh, "Sa": Sa, "Sf": Sf})
    st.caption(
        f"Ağırlıklar → Sv:{w['v']:.2f} | Sl:{w['l']:.2f} | Sb:{w['b']:.2f} | "
        f"Sh:{w['h']:.2f} | Sa:{w['a']:.2f}"
    )

    if show_timeline:
        with st.expander("📈 Sf zaman grafiği", expanded=True):
            render_sf_timeline(frame_scores, fps=fps)

# =========================================================
# EN ŞÜPHELİ KARELER (3 FAZ BİRLİKTE)
# =========================================================
st.markdown("---")
st.subheader("🔥 En Şüpheli Kareler (Explainability)")

if frame_scores:
    top = sorted(frame_scores, key=lambda x: _safe(x[1]), reverse=True)
    cols = st.columns(5, gap="small")

    pdf_frames: List[Tuple[str, float, str, str, np.ndarray]] = []

    for i, item in enumerate(top):
        img_path = item[0]
        s = item[1]
        score = _clip01(s)
        frame_idx = _parse_frame_idx(img_path, i)
        tc = frame_to_timecode(frame_idx, fps)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cam_img, cam_kind = combine_cam_and_landmarks(
            img_rgb,
            prefer_real_cam=prefer_real_cam,
            alpha_cam=alpha_cam,
            show_landmarks=show_landmarks,
        )

        with cols[i]:
            st.image(cam_img, use_container_width=True)
            st.caption(f"p(fake)={score:.2f} · ⏱ {tc}")
            st.caption(frame_explain(score))

            with st.expander("Detay / Neden?"):
                st.write(f"**CAM türü**: {cam_kind}")
                st.write(f"**Mod**: {'Konuşmalı' if has_speech else 'Sessiz'}")
                st.write(f"**Toplam Sf**: `{Sf:.3f}`")

        pdf_frames.append((img_path, score, tc, cam_kind, cam_img))

# =====================================================
# VERDICT (NIHAI KARAR MANTIGI)
# =====================================================

def compute_verdict(Sf, Sv, Sa):
    """
    Basit ama güvenli karar mantığı
    """
    if Sf >= 0.75 and Sv >= 0.65:
        return "DEEPFAKE"
    elif Sa >= 0.60 and Sf < 0.75:
        return "AI_GENERATED / COMPRESSION"
    elif Sf < 0.40 and Sv < 0.40:
        return "REAL"
    else:
        return "UNCERTAIN"


verdict = compute_verdict(Sf, Sv, Sa)


# =====================================================
# NIHAYI KARAR (PDF'DEN ÖNCE GÖSTERİLMELİ)
# =====================================================
st.subheader("🧠 Nihai Karar")

if verdict == "DEEPFAKE":
    st.error("⚠️ Deepfake tespit edildi")
elif verdict == "AI_GENERATED / COMPRESSION":
    st.warning("🤖 AI üretimi / sıkıştırma artefaktı (deepfake değil)")
elif verdict == "REAL":
    st.success("✅ Gerçek video")
else:
    st.info("🟡 Şüpheli – manuel inceleme önerilir")

# PDF için güvenli frame listesi
if "top_frames" in locals() and top_frames:
    pdf_frames = [
        (img_path, s, t, "Model highlight")
        for (img_path, s, t) in top_frames
    ]
else:
    pdf_frames = []  # <<< KRİTİK SATIR


# =====================================================
# PDF
# =====================================================
st.markdown("---")
st.subheader("📄 Rapor (PDF)")

pdf_bytes = generate_pdf_report(
    VIDEO_TMP,
    has_speech,
    text,
    {
        "Sv": Sv,
        "Sl": Sl,
        "Sb": Sb,
        "Sh": Sh,
        "Sa": Sa,
        "Sf": Sf,
        "Verdict": verdict,   # 👈 ÖNEMLİ
    },
    w,
    fps,
    pdf_frames,
)

if pdf_bytes:
    st.download_button(
        "⬇ PDF Raporu indir",
        data=pdf_bytes,
        file_name="deepfake_report.pdf",
        mime="application/pdf",
    )
else:
    st.info("PDF için reportlab kurulu değil (opsiyonel).")


# =========================================================
# ÇALIŞTIRMA
# =========================================================
# .\.venv\Scripts\Activate.ps1
# python -m streamlit run src/app.py
