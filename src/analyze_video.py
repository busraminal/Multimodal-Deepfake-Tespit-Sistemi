import argparse
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

from src.audio_artefact import audio_gan_score
from src.fusion import interpret_score
from src.visual_score import visual_score


def _clip01(x) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _fusion_v3(sv: float, sl: float, sb: float, sh: float, sa: float, has_speech: bool):
    if has_speech:
        w = {"v": 0.45, "l": 0.20, "b": 0.10, "h": 0.10, "a": 0.15}
    else:
        w = {"v": 0.55, "l": 0.00, "b": 0.15, "h": 0.10, "a": 0.20}
    sf = (w["v"] * sv) + (w["l"] * sl) + (w["b"] * sb) + (w["h"] * sh) + (w["a"] * sa)
    return _clip01(sf), w


def _extract_audio_basic(video_path: str, audio_path: str) -> None:
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        audio_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # ffmpeg yoksa pipeline'ı düşürmeyelim: 1sn sessiz wav üret.
        with wave.open(audio_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)


def _extract_frames_basic(video_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        y1 = int(h * 0.40)
        y2 = int(h * 0.92)
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            roi = frame
        roi = cv2.resize(roi, (96, 96))
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.jpg"), roi)
        i += 1
    cap.release()


def analyze(video_path: str):
    video_path = str(Path(video_path).resolve())
    if not Path(video_path).exists():
        raise FileNotFoundError(video_path)

    with tempfile.TemporaryDirectory(prefix="df_") as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        used_fallback = False
        pipeline_error = ""
        try:
            from src.media_io import extract_audio, extract_frames

            extract_audio(video_path, audio_path)
            extract_frames(video_path, frames_dir)
        except Exception as exc:
            used_fallback = True
            pipeline_error = str(exc)
            _extract_audio_basic(video_path, audio_path)
            _extract_frames_basic(video_path, frames_dir)

        sv, top_frames = visual_score(frames_dir, topk=5, save_gradcam=False)
        try:
            from src.biomech import blink_score, headpose_score

            sb = blink_score(frames_dir)
            sh = headpose_score(frames_dir)
        except Exception:
            sb = 0.5
            sh = 0.5
        sa, audio_details = audio_gan_score(audio_path, True)

        # CLI için hızlı mod: transkripsiyon açmadan lip-sync default pas
        has_speech = False
        sl = 0.0
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 2048:
            try:
                from src.lip_sync import lip_mismatch_score

                sl = lip_mismatch_score(audio_path, frames_dir)
                has_speech = True
            except Exception:
                sl = 0.0
                has_speech = False

        sv, sl, sb, sh, sa = map(_clip01, (sv, sl, sb, sh, sa))
        sf, weights = _fusion_v3(sv, sl, sb, sh, sa, has_speech)
        verdict, verdict_msg = interpret_score(sf, has_speech=has_speech)

        return {
            "video_path": video_path,
            "scores": {"Sv": sv, "Sl": sl, "Sb": sb, "Sh": sh, "Sa": sa, "Sf": sf},
            "weights": weights,
            "has_speech": has_speech,
            "verdict": verdict,
            "verdict_message": verdict_msg,
            "top_suspicious_frames": top_frames,
            "audio_details": audio_details,
            "used_fallback_pipeline": used_fallback,
            "pipeline_error": pipeline_error,
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze a video for deepfake risk.")
    parser.add_argument("--video", required=True, help="Input mp4 video path")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    result = analyze(args.video)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
