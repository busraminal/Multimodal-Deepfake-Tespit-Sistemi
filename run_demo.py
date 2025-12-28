# run_demo.py
import os

# FFmpeg yolu (sen zaten eklemiştin, burada kalsın)
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ["PATH"]

from src.media_io import extract_audio, extract_frames
from src.asr_text import transcribe
from src.lip_sync import lip_mismatch_score
from src.visual_score import visual_score_dummy  # HF modeline geçtiysen burayı değiştirirsin
from src.fusion import compute_scores, interpret_score


# Test için kullandığın video
VIDEO_PATH = r"C:\Users\Casper\Desktop\deepfake_project\data\videos\example.mp4"


def main():
    audio_path = "data/audio/example.wav"
    frames_dir = "data/frames/mouth"

    print("\n[1] Extracting audio...")
    extract_audio(VIDEO_PATH, audio_path)

    print("[2] Extracting mouth frames...")
    extract_frames(VIDEO_PATH, frames_dir)

    print("[3] Running ASR (Whisper)...")
    text = transcribe(audio_path, language="tr")

    print("[4] Calculating lip-sync (signal-based Sl)...")
    Sl_signal = lip_mismatch_score(audio_path, frames_dir)
    print(f"    Sl_signal: {Sl_signal:.3f}")

    print("[5] Visual scoring (deepfake model / dummy)...")
    Sv_raw = visual_score_dummy(frames_dir)   # HF modeline geçtiğinde burada gerçek fonksiyonu çağır

    print("[6] Fusion scoring...")
    Sv, Sl, Sf = compute_scores(Sv_raw, Sl_signal, w_visual=0.7)
    comment = interpret_score(Sf)

    print("\n==================== RESULTS ====================")
    print(f"Transcript:  {text}")
    print(f"Sv (visual score)       : {Sv:.3f}")
    print(f"Sl (lip mismatch score) : {Sl:.3f}")
    print(f"Sf (final fusion score) : {Sf:.3f}")
    print(f"Yorum                   : {comment}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
