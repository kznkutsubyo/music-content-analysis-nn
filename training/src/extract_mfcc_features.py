from __future__ import annotations
from pathlib import Path
import argparse, csv, json, random
import numpy as np
import soundfile as sf

import torch
import torchaudio
from torchaudio.transforms import Resample

def load_csv(csv_path: Path):
    items = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append((row["path"], row["genre"]))
    return items

def read_audio_soundfile(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    audio, sr0 = sf.read(str(p), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise RuntimeError(f"Empty audio file: {p}")

    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)

    y = torch.from_numpy(audio.T)
    return y, int(sr0)

def crop_or_pad(y: torch.Tensor, sr: int, crop_sec: float, train: bool, seed: int, idx: int):
    if crop_sec <= 0:
        return y
    crop_len = int(sr * crop_sec)
    T = y.shape[1]

    if T < crop_len:
        return torch.nn.functional.pad(y, (0, crop_len - T))
    if T == crop_len:
        return y

    if train:
        rng = random.Random(seed + idx)
        start = rng.randint(0, T - crop_len)
    else:
        start = (T - crop_len) // 2

    return y[:, start:start + crop_len]

def mfcc_stats(y: torch.Tensor, sr: int, num_ceps: int = 20, num_mel_bins: int = 40):
    mfcc = torchaudio.compliance.kaldi.mfcc(
        y,
        sample_frequency=sr,
        num_ceps=num_ceps,
        num_mel_bins=num_mel_bins,
        frame_shift=10,
        use_energy=False,
        dither=0.0,
        window_type="hanning",
    )

    m = mfcc.mean(dim=0)
    s = mfcc.std(dim=0, unbiased=False)
    feat = torch.cat([m, s], dim=0)
    return feat.numpy().astype(np.float32)

def extract_split(items, label2idx, target_sr: int, crop_sec: float, train: bool, seed: int):
    X, y, paths = [], [], []

    resamplers: dict[int, Resample] = {}

    for i, (path, genre) in enumerate(items):
        try:
            wav, sr0 = read_audio_soundfile(path)
            if sr0 != target_sr:
                r = resamplers.get(sr0)
                if r is None:
                    r = Resample(sr0, target_sr)
                    resamplers[sr0] = r
                wav = r(wav)

            wav = crop_or_pad(wav, target_sr, crop_sec=crop_sec, train=train, seed=seed, idx=i)
            feat = mfcc_stats(wav, target_sr)

            X.append(feat)
            y.append(label2idx[genre])
            paths.append(path)
        except Exception as e:
            raise RuntimeError(f"Failed on file: {path}\n{e}") from e

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y, np.array(paths)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--crop_sec", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label2idx = json.loads((splits_dir / "label2idx.json").read_text(encoding="utf-8"))
    idx2label = {int(v): k for k, v in label2idx.items()}

    (out_dir / "label2idx.json").write_text(json.dumps(label2idx, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "idx2label.json").write_text(json.dumps(idx2label, indent=2, ensure_ascii=False), encoding="utf-8")

    train_items = load_csv(splits_dir / "train.csv")
    val_items   = load_csv(splits_dir / "val.csv")
    test_items  = load_csv(splits_dir / "test.csv")

    print("Extracting TRAIN...")
    Xtr, ytr, ptr = extract_split(train_items, label2idx, args.sr, args.crop_sec, train=True, seed=args.seed)
    np.savez(out_dir / "train_mfcc.npz", X=Xtr, y=ytr, paths=ptr)

    print("Extracting VAL...")
    Xva, yva, pva = extract_split(val_items, label2idx, args.sr, args.crop_sec, train=False, seed=args.seed)
    np.savez(out_dir / "val_mfcc.npz", X=Xva, y=yva, paths=pva)

    print("Extracting TEST...")
    Xte, yte, pte = extract_split(test_items, label2idx, args.sr, args.crop_sec, train=False, seed=args.seed)
    np.savez(out_dir / "test_mfcc.npz", X=Xte, y=yte, paths=pte)

    print("Done.")
    print("TRAIN:", Xtr.shape, ytr.shape)
    print("VAL  :", Xva.shape, yva.shape)
    print("TEST :", Xte.shape, yte.shape)

if __name__ == "__main__":
    main()
