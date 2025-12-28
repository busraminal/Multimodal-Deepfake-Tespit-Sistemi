import whisper
from sentence_transformers import SentenceTransformer
import numpy as np

_whisper_model = None
_txt_model = None

def get_whisper(model_name="medium"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model

def get_txt_model():
    global _txt_model
    if _txt_model is None:
        _txt_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _txt_model

def transcribe(audio_path, language="tr"):
    model = get_whisper()
    result = model.transcribe(audio_path, language=language)
    return result["text"]

def text_embed(text: str):
    model = get_txt_model()
    v = model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(v, dtype=np.float32)
