import requests

LLM_URL = "https://rear-representation-dear-nowhere.trycloudflare.com/analyze"

def send_to_llm(video_id, features):
    try:
        r = requests.post(
            LLM_URL,
            json={
                "video_id": video_id,
                "features": features
            },
            timeout=5   #  MUTLAKA
        )
        r.raise_for_status()
        return r.json()

    except Exception as e:
        return {
            "confidence": max(features),
            "explanation": (
                "LLM bağlantısı gecikti veya geçici olarak erişilemedi. "
                "Lokal multimodal skorlar kullanıldı."
            )
        }
