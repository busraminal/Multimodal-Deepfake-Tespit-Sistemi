# -----------------------------------------------------------
# media_io.py  (GÜNCEL • HATASIZ • FFmpeg + FaceMesh)
# -----------------------------------------------------------

import mediapipe as mp
mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



import cv2
import ffmpeg
import mediapipe as mp
import os
import shutil
import numpy as np


def _resolve_ffmpeg_exe() -> str:
    for key in ("FFMPEG_PATH", "FFMPEG_BIN"):
        p = (os.environ.get(key) or "").strip().strip('"')
        if p and os.path.isfile(p):
            return p
    win = r"C:\ffmpeg\bin\ffmpeg.exe"
    if os.path.isfile(win):
        return win
    w = shutil.which("ffmpeg")
    if w:
        return w
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


FFMPEG_PATH = _resolve_ffmpeg_exe()
FFMPEG_QUIET = os.environ.get("FFMPEG_QUIET", "0").lower() in ("1", "true", "yes")

# MediaPipe yüz mesh modeli
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Ağız landmark indexleri
MOUTH_LANDMARKS = list(range(61, 88))


# -----------------------------------------------------------
# 1) SES ÇIKARTMA – Whisper uyumlu WAV üretir
# -----------------------------------------------------------
def extract_audio(video_path, audio_path):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    try:
        (
            ffmpeg
            .input(video_path)
            .output(
                audio_path,
                acodec="pcm_s16le",
                ac=1,
                ar=16000
            )
            .overwrite_output()
            .run(cmd=FFMPEG_PATH, quiet=FFMPEG_QUIET)
        )
    except ffmpeg.Error as e:
        print("FFMPEG HATASI:")
        print(e.stderr.decode() if e.stderr else e)
        raise


def safe_silent_wav(audio_path):
    import wave
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    with wave.open(audio_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b'\x00\x00' * 16000)   # 1 saniye sessiz
    print("[WARN] silent video; created empty WAV fallback.")


def extract_audio(video_path, audio_path):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec="pcm_s16le", ac=1, ar="16000")
            .overwrite_output()
            .run(cmd=FFMPEG_PATH, quiet=FFMPEG_QUIET)
        )
    except ffmpeg.Error as e:
        print("FFmpeg hata verdi, fallback çalışıyor…")
        print(e.stderr.decode() if e.stderr else e)
        safe_silent_wav(audio_path)


# -----------------------------------------------------------
# 2) FRAME ÇIKARMA – Ağız ROI 96×96 çıkarır
# -----------------------------------------------------------
def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(frame)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]

            h, w, _ = frame.shape

            xs, ys = [], []

            # ağzı bul
            for idx in MOUTH_LANDMARKS:
                x = int(lm.landmark[idx].x * w)
                y = int(lm.landmark[idx].y * h)
                xs.append(x)
                ys.append(y)

            min_x = max(min(xs) - 15, 0)
            max_x = min(max(xs) + 15, w)
            min_y = max(min(ys) - 15, 0)
            max_y = min(max(ys) + 15, h)

            mouth_crop = frame[min_y:max_y, min_x:max_x]

            # 96×96 normalize
            mouth_crop = cv2.resize(mouth_crop, (96, 96))

            out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2BGR))

        frame_idx += 1

    cap.release()
    print("[OK] mouth frames saved:", out_dir)
