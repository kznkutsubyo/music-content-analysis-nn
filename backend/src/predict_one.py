from __future__ import annotations
from pathlib import Path
import json
import torch
import torchaudio
from torchaudio.transforms import Resample
from models import ASTModel
import torchaudio.compliance.kaldi as kaldi
import random

NORM_ADD = 4.26
NORM_DIV = 4.57 * 2.0
TDIM = 1024
SR = 16000

def to_fbank(wav: torch.Tensor, sr: int):
    fb = kaldi.fbank(
        wav,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )
    T = fb.shape[0]
    if T < TDIM:
        fb = torch.nn.functional.pad(fb, (0, 0, 0, TDIM - T))
    elif T > TDIM:
        start = (T - TDIM) // 2
        fb = fb[start:start + TDIM]
    fb = (fb + NORM_ADD) / NORM_DIV
    return fb

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(Path("..") / "gtzan_ast_best.pt"))
    ap.add_argument("--wav", type=str, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    label2idx = ckpt["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}

    model = ASTModel(label_dim=len(label2idx), input_tdim=TDIM, imagenet_pretrain=False, audioset_pretrain=False, model_size="base384")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    y, sr0 = torchaudio.load(args.wav)
    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
    if sr0 != SR:
        y = Resample(sr0, SR)(y)

    crop_len = int(SR * 10.0)
    if y.shape[1] < crop_len:
        y = torch.nn.functional.pad(y, (0, crop_len - y.shape[1]))
    elif y.shape[1] > crop_len:
        start = (y.shape[1] - crop_len) // 2
        y = y[:, start:start + crop_len]

    fb = to_fbank(y, SR).unsqueeze(0)

    with torch.no_grad():
        logits = model(fb)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top = torch.topk(probs, k=min(5, probs.numel()))
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"{idx2label[i]:10s} {p:.4f}")

if __name__ == "__main__":
    main()
