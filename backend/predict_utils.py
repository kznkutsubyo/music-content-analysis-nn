from __future__ import annotations

import sys
import json
import math
import subprocess
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn

import torchaudio
from torchaudio.transforms import Resample

from joblib import load as joblib_load


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import ASTModel


NORM_ADD = 4.26
NORM_DIV = 4.57 * 2.0


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.head(x)


def _read_label2idx(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def _idx2label(label2idx: Dict[str, int]) -> List[str]:
    idx2 = {int(v): k for k, v in label2idx.items()}
    return [idx2[i] for i in range(len(idx2))]


def _soundfile_read(path: Path) -> Tuple[torch.Tensor, int]:
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if audio.size == 0:
            raise RuntimeError("Empty audio file")
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        y = torch.from_numpy(audio.T)
        return y, int(sr)
    except Exception as soundfile_error:
        try:
            y, sr = torchaudio.load(str(path))
        except Exception as torchaudio_error:
            ffmpeg_error_msg = "not attempted"
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as converted:
                    converted_path = Path(converted.name)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(path),
                        "-ac",
                        "1",
                        str(converted_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                try:
                    return _soundfile_read(converted_path)
                finally:
                    converted_path.unlink(missing_ok=True)
            except Exception as ffmpeg_error:
                ffmpeg_error_msg = str(ffmpeg_error)
                pass

            raise RuntimeError(
                f"Could not decode audio with soundfile ({soundfile_error}) "
                f"or torchaudio ({torchaudio_error}); ffmpeg fallback failed ({ffmpeg_error_msg})"
            ) from torchaudio_error

        if y.numel() == 0:
            raise RuntimeError("Empty audio file")
        if y.shape[0] > 1:
            y = y.mean(dim=0, keepdim=True)
        return y.to(torch.float32), int(sr)


def _center_crop_or_pad(y: torch.Tensor, sr: int, crop_sec: float) -> torch.Tensor:
    if crop_sec <= 0:
        return y
    crop_len = int(sr * crop_sec)
    T = y.shape[1]
    if T < crop_len:
        return torch.nn.functional.pad(y, (0, crop_len - T))
    if T == crop_len:
        return y
    start = (T - crop_len) // 2
    return y[:, start:start + crop_len]


def _to_fbank(y: torch.Tensor, sr: int, tdim: int) -> torch.Tensor:
    fb = torchaudio.compliance.kaldi.fbank(
        y,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    T = fb.shape[0]
    if T < tdim:
        fb = torch.nn.functional.pad(fb, (0, 0, 0, tdim - T))
    elif T > tdim:
        start = (T - tdim) // 2
        fb = fb[start:start + tdim]

    fb = (fb + NORM_ADD) / NORM_DIV
    return fb


def _mfcc_stats(y: torch.Tensor, sr: int) -> np.ndarray:
    mfcc = torchaudio.compliance.kaldi.mfcc(
        y,
        sample_frequency=sr,
        num_ceps=20,
        num_mel_bins=40,
        frame_shift=10,
        use_energy=False,
        dither=0.0,
        window_type="hanning",
    )

    m = mfcc.mean(dim=0)
    s = mfcc.std(dim=0, unbiased=False)
    feat = torch.cat([m, s], dim=0)
    return feat.numpy().astype(np.float32)


def _topk(proba: np.ndarray, labels: List[str], k: int = 3) -> Dict[str, Any]:
    k = min(k, proba.shape[-1])
    idx = np.argsort(-proba)[:k]
    out = [{"label": labels[int(i)], "prob": float(proba[int(i)])} for i in idx]
    return {"top1": out[0], "topk": out}


def _cuda_autocast_enabled(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


@dataclass
class PredictorConfig:
    splits_dir: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    ast_ckpt: Path = ROOT / "gtzan_ast_best.pt"
    ast_model_size: str = "base384"
    ast_tdim: int = 1024
    ast_crop_sec: float = 10.0

    cnn_ckpt: Path = ROOT / "artifacts" / "cnn_fbank_best.pt"
    cnn_tdim: int = 1024
    cnn_crop_sec: float = 10.0

    knn_path: Path = ROOT / "artifacts" / "knn_mfcc.joblib"
    rf_path: Path = ROOT / "artifacts" / "rf_mfcc.joblib"
    mfcc_crop_sec: float = 30.0

    target_sr: int = 16000


class Predictor:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        label2idx_path = cfg.splits_dir / "label2idx.json"
        self.label2idx = _read_label2idx(label2idx_path)
        self.labels = _idx2label(self.label2idx)
        self.n_classes = len(self.labels)

        self._resamplers: Dict[int, Resample] = {}

        self.ast_model: Optional[nn.Module] = None
        self.cnn_model: Optional[nn.Module] = None
        self.knn = None
        self.rf = None

        self._load_models()

    def _get_resampler(self, sr0: int) -> Resample:
        r = self._resamplers.get(sr0)
        if r is None:
            r = Resample(sr0, self.cfg.target_sr)
            self._resamplers[sr0] = r
        return r

    def _load_models(self):
        if self.cfg.ast_ckpt.exists():
            ckpt = torch.load(self.cfg.ast_ckpt, map_location=self.device)
            m = ASTModel(
                label_dim=self.n_classes,
                input_tdim=self.cfg.ast_tdim,
                imagenet_pretrain=False,
                audioset_pretrain=False,
                model_size=self.cfg.ast_model_size,
            ).to(self.device)
            m.load_state_dict(ckpt["model_state"], strict=True)
            m.eval()
            self.ast_model = m

        if self.cfg.cnn_ckpt.exists():
            ckpt = torch.load(self.cfg.cnn_ckpt, map_location=self.device)
            m = SmallCNN(n_classes=self.n_classes).to(self.device)
            m.load_state_dict(ckpt["model_state"], strict=True)
            m.eval()
            self.cnn_model = m

        if self.cfg.knn_path.exists():
            self.knn = joblib_load(self.cfg.knn_path)
        if self.cfg.rf_path.exists():
            self.rf = joblib_load(self.cfg.rf_path)

    def _preprocess_audio(self, tmp_path: Path) -> Tuple[torch.Tensor, int, float]:
        y, sr0 = _soundfile_read(tmp_path)
        dur = y.shape[1] / float(sr0)

        if sr0 != self.cfg.target_sr:
            y = self._get_resampler(sr0)(y)
            sr0 = self.cfg.target_sr

        return y, sr0, dur

    def _predict_ast(self, y: torch.Tensor, sr: int) -> Dict[str, Any]:
        assert self.ast_model is not None
        y10 = _center_crop_or_pad(y, sr, self.cfg.ast_crop_sec)
        fb = _to_fbank(y10, sr, self.cfg.ast_tdim)
        xb = fb.unsqueeze(0).to(self.device, non_blocking=True)

        with torch.inference_mode():
            with _cuda_autocast_enabled(self.device):
                logits = self.ast_model(xb)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return _topk(prob, self.labels, k=3)

    def _predict_cnn(self, y: torch.Tensor, sr: int) -> Dict[str, Any]:
        assert self.cnn_model is not None
        y10 = _center_crop_or_pad(y, sr, self.cfg.cnn_crop_sec)
        fb = _to_fbank(y10, sr, self.cfg.cnn_tdim)
        x = fb.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        x = x.to(self.device, non_blocking=True)

        with torch.inference_mode():
            with _cuda_autocast_enabled(self.device):
                logits = self.cnn_model(x)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return _topk(prob, self.labels, k=3)

    def _predict_mfcc_model(self, model, y: torch.Tensor, sr: int) -> Dict[str, Any]:
        y30 = _center_crop_or_pad(y, sr, self.cfg.mfcc_crop_sec)
        feat = _mfcc_stats(y30, sr).reshape(1, -1)
        proba = model.predict_proba(feat)[0]
        return _topk(proba, self.labels, k=3)

    def predict_from_bytes(self, data: bytes, filename: str) -> Dict[str, Any]:
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            tmp = Path(f.name)
            f.write(data)

        try:
            y, sr, dur = self._preprocess_audio(tmp)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

        out: Dict[str, Any] = {
            "filename": filename,
            "duration_sec": float(dur),
            "sr": int(sr),
            "predictions": {}
        }

        if self.ast_model is not None:
            out["predictions"]["AST"] = self._predict_ast(y, sr)
        else:
            out["predictions"]["AST"] = {"error": "AST checkpoint not found"}

        if self.cnn_model is not None:
            out["predictions"]["CNN"] = self._predict_cnn(y, sr)
        else:
            out["predictions"]["CNN"] = {"error": "CNN checkpoint not found"}

        if self.rf is not None:
            out["predictions"]["RandomForest"] = self._predict_mfcc_model(self.rf, y, sr)
        else:
            out["predictions"]["RandomForest"] = {"error": "RF model not found"}

        if self.knn is not None:
            out["predictions"]["KNN"] = self._predict_mfcc_model(self.knn, y, sr)
        else:
            out["predictions"]["KNN"] = {"error": "KNN model not found"}

        return out
