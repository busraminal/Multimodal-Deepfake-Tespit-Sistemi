import os
import numpy as np
import librosa
import cv2
import mediapipe as mp
from scipy.spatial.distance import euclidean

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


# -----------------------------------------------------
# 3) Korelasyon (lip-sync senkronu)
# -----------------------------------------------------
def lip_sync_correlation(audio_energy, lip_motion, max_lag=6):
    # ikisini aynı uzunluğa getir
    L = min(len(audio_energy), len(lip_motion))
    a = audio_energy[:L]
    v = lip_motion[:L]
    if L < 4:
        return 0.5

    # Farkli AV gecikmelerine toleransli sync olcumu.
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
        # anti-sync de uyumsuzluk göstergesi olabildiği için mutlak korelasyon.
        best = max(best, abs(float(c)))

    if best < 0:
        best = 0.0
    return float(np.clip(best, 0.0, 1.0))


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
