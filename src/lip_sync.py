import os
import numpy as np
import librosa
import cv2
import mediapipe as mp
from scipy.spatial.distance import euclidean

# Ses–dudak ölçümü iyileştirildiğinde feature_cache yenilenmeli (refresh_sl_cache / tam analiz).
_LIP_SYNC_SMOOTH_WIN = max(3, int(os.environ.get("DF_LIP_SYNC_SMOOTH_WIN", "5")))
_VELOCITY_BLEND = float(os.environ.get("DF_LIP_SYNC_VELOCITY_BLEND", "0.45"))
_MIN_LIP_VAR = float(os.environ.get("DF_LIP_SYNC_MIN_VAR", "1e-4"))
_MIN_AUDIO_VAR = float(os.environ.get("DF_LIP_SYNC_MIN_AUDIO_VAR", "1e-8"))


def has_speech_like_audio(
    audio_path: str,
    *,
    sr: int = 16000,
    frame_ms: float = 25.0,
    hop_frac: float = 0.5,
    min_duration_sec: float = 0.25,
    min_active_frac: float = 0.06,
) -> bool:
    """
    Hafif enerji/VAD: konuşma benzeri aktivite yoksa lip-sync skorunu üretmeyelim
    (sessiz/zayıf ses/gürültüde korelasyon Sahte Sl üretmesin).

    Cache/fusion bu mantıkla değiştiği için özellik önbelleğini yeniden üretmek gerekir.
    """
    wav, _sr = librosa.load(audio_path, sr=sr, mono=True)
    if wav.size < int(sr * min_duration_sec):
        return False
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(frame * hop_frac))
    rms_list = []
    for i in range(0, len(wav) - frame + 1, hop):
        chunk = wav[i : i + frame]
        rms_list.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))))
    if len(rms_list) < 3:
        return False
    rms = np.array(rms_list, dtype=np.float64)
    floor = float(np.percentile(rms, 25))
    thresh = max(floor * 4.0, 1e-5)
    active = rms > thresh
    return float(np.mean(active.astype(np.float64))) >= min_active_frac

# =====================================================
# 1) Dudak landmarkları + audio enerji analizi
# =====================================================

mp_face = mp.solutions.face_mesh
FACE_MESH = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

# Dudak landmark indeksleri (mediapipe)
LIPS = list(range(61, 88))  # Outer + inner lips


# -----------------------------------------------------
# 1) Audiodan enerji sinyali çıkar (frame-level)
# -----------------------------------------------------
def extract_audio_energy(audio_path, fps=25):
    wav, sr = librosa.load(audio_path, sr=16000)

    frame_size = int(sr / fps)
    energies = []

    for i in range(0, len(wav), frame_size):
        frame = wav[i:i + frame_size]
        if len(frame) == 0:
            continue
        energy = np.sum(frame ** 2)
        energies.append(energy)

    energies = np.array(energies)
    if len(energies) == 0:
        return None

    # normalize 0-1 (range-based)
    e_min = float(energies.min())
    e_max = float(energies.max())
    energies = (energies - e_min) / (e_max - e_min + 1e-9)
    return energies


# -----------------------------------------------------
# 2) Videodan dudak açılma miktarı
# -----------------------------------------------------
def lip_openings(frames_dir, fps=25):
    mouth_open_vals = []

    for fname in sorted(os.listdir(frames_dir)):
        if not fname.endswith(".png") and not fname.endswith(".jpg"):
            continue

        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = FACE_MESH.process(rgb)

        if not result.multi_face_landmarks:
            mouth_open_vals.append(0)
            continue

        pts = result.multi_face_landmarks[0].landmark

        # Dudak üstü (13) ve dudak altı (14) arası dikey mesafe
        top = pts[13]
        bottom = pts[14]

        h = euclidean((top.x, top.y), (bottom.x, bottom.y))
        mouth_open_vals.append(h)

    arr = np.array(mouth_open_vals)
    if len(arr) == 0:
        return None

    # normalize 0-1 (range-based)
    a_min = float(arr.min())
    a_max = float(arr.max())
    arr = (arr - a_min) / (a_max - a_min + 1e-9)
    return arr


def _smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if win <= 1 or len(x) < 3:
        return x
    win = min(win | 1, len(x))
    k = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, k, mode="same")


def _lagged_best_abs_corr(a: np.ndarray, v: np.ndarray, max_lag: int) -> float:
    L = min(len(a), len(v))
    a = a[:L]
    v = v[:L]
    if L < 4:
        return 0.5
    best = -1.0
    max_lag = int(max(0, min(max_lag, L // 3)))
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[:lag]
            vv = v[-lag:]
        elif lag > 0:
            aa = a[lag:]
            vv = v[:-lag]
        else:
            aa = a
            vv = v
        if len(aa) < 4:
            continue
        c = np.corrcoef(aa, vv)[0, 1]
        if np.isnan(c):
            c = 0.0
        best = max(best, abs(float(c)))
    if best < 0:
        best = 0.0
    return float(np.clip(best, 0.0, 1.0))


# -----------------------------------------------------
# 3) Korelasyon (lip-sync senkronu)
# -----------------------------------------------------
def lip_sync_correlation(audio_energy, lip_motion, max_lag=6):
    """
    Ses enerji zarfı ile dudak açılımı serisinin (lag ile) mutlak korelasyon tepe değeri.
    Çok düşük varyans veya statik dudakta güven düşük → ~0.5 (belirsiz).
    """
    L = min(len(audio_energy), len(lip_motion))
    a = np.asarray(audio_energy[:L], dtype=np.float64)
    v = np.asarray(lip_motion[:L], dtype=np.float64)
    if L < 4:
        return 0.5

    if float(np.var(a)) < _MIN_AUDIO_VAR or float(np.var(v)) < _MIN_LIP_VAR:
        return 0.5

    a = _smooth_1d(a, _LIP_SYNC_SMOOTH_WIN)
    v = _smooth_1d(v, _LIP_SYNC_SMOOTH_WIN)

    sync_raw = _lagged_best_abs_corr(a, v, max_lag=max_lag)

    da = np.diff(a)
    dv = np.diff(v)
    w = float(np.clip(_VELOCITY_BLEND, 0.0, 1.0))
    if len(da) >= 4 and len(dv) >= 4 and w > 1e-6:
        sync_vel = _lagged_best_abs_corr(da, dv, max_lag=max(2, max_lag - 1))
        sync_score = (1.0 - w) * sync_raw + w * sync_vel
    else:
        sync_score = sync_raw

    return float(np.clip(sync_score, 0.0, 1.0))


# -----------------------------------------------------
# 4) Ana fonksiyon (Sl_signal)
# -----------------------------------------------------
def lip_mismatch_score(audio_path, frames_dir):
    audio_energy = extract_audio_energy(audio_path)
    lip_motion = lip_openings(frames_dir)

    if audio_energy is None or lip_motion is None:
        return 0.0

    sync_score = lip_sync_correlation(audio_energy, lip_motion)
    # Fusion'da Sl fake yönlü kullanılıyor: yüksek Sl => daha çok mismatch.
    mismatch_score = 1.0 - sync_score
    return float(np.clip(mismatch_score, 0.0, 1.0))
