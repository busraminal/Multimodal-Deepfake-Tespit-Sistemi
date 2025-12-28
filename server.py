from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class AnalyzeReq(BaseModel):
    video_id: str
    scores: List[float]
    question: Optional[str] = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    question = req.question or "Bu video neden gerçek sayıldı?"

    return {
        "video_id": req.video_id,
        "answer": (
            f"Soru: {question}\n"
            f"Skorlar: {req.scores}\n"
            "Sonuç: Skorlar gerçek videolarla daha uyumludur."
        )
    }
