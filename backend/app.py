from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict_utils import Predictor, PredictorConfig, ROOT


app = FastAPI(title="GTZAN Genre Classification API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTOR: Optional[Predictor] = None


@app.on_event("startup")
def _startup():
    global PREDICTOR
    splits_dir = ROOT / "gtzan_splits"
    if not splits_dir.exists():
        raise RuntimeError(f"Splits dir not found: {splits_dir}")

    device = os.getenv("MODEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PredictorConfig(splits_dir=splits_dir, device=device)
    PREDICTOR = Predictor(cfg)


@app.get("/health")
def health() -> Dict[str, Any]:
    if PREDICTOR is None:
        return {"ok": False, "error": "Predictor not loaded"}

    cfg = PREDICTOR.cfg
    return {
        "ok": True,
        "device": cfg.device,
        "splits_dir": str(cfg.splits_dir),
        "models": {
            "ast_ckpt_exists": cfg.ast_ckpt.exists(),
            "cnn_ckpt_exists": cfg.cnn_ckpt.exists(),
            "rf_exists": cfg.rf_path.exists(),
            "knn_exists": cfg.knn_path.exists(),
        }
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    p = ROOT / "web_results.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="web_results.json not found")
    import json
    return json.loads(p.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if PREDICTOR is None:
        raise HTTPException(status_code=500, detail="Predictor not loaded")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = PREDICTOR.predict_from_bytes(data, file.filename or "audio.wav")
        return result
    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Audio decode failed: {e}. Try WAV, MP3, FLAC, OGG, AU, AIFF, M4A/AAC."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
