from __future__ import annotations
from pathlib import Path
import argparse, csv, json, random

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.transforms import Resample

NORM_ADD = 4.26
NORM_DIV = 4.57 * 2.0

def load_csv(csv_path: Path):
    items = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append((row["path"], row["genre"]))
    return items

class GTZANFbankDataset(Dataset):
    def __init__(self, items, label2idx, train: bool, sr: int = 16000, crop_sec: float = 10.0, tdim: int = 1024, seed: int = 42):
        self.items = items
        self.label2idx = label2idx
        self.train = train
        self.sr = sr
        self.crop_sec = crop_sec
        self.tdim = tdim
        self.seed = seed
        self._resamplers: dict[int, Resample] = {}

    def __len__(self):
        return len(self.items)

    def _read_audio(self, path: str):
        audio, sr0 = sf.read(path, dtype="float32", always_2d=True)
        if audio.size == 0:
            raise RuntimeError(f"Empty audio: {path}")
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        y = torch.from_numpy(audio.T)
        return y, int(sr0)

    def _crop(self, y: torch.Tensor, idx: int):
        crop_len = int(self.sr * self.crop_sec)
        T = y.shape[1]
        if T < crop_len:
            return torch.nn.functional.pad(y, (0, crop_len - T))
        if T == crop_len:
            return y
        if self.train:
            rng = random.Random(self.seed + idx)
            start = rng.randint(0, T - crop_len)
        else:
            start = (T - crop_len) // 2
        return y[:, start:start + crop_len]

    def _to_fbank(self, wav: torch.Tensor):
        fb = torchaudio.compliance.kaldi.fbank(
            wav,
            htk_compat=True,
            sample_frequency=self.sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )

        T = fb.shape[0]
        if T < self.tdim:
            fb = torch.nn.functional.pad(fb, (0, 0, 0, self.tdim - T))
        elif T > self.tdim:
            if self.train:
                rng = random.Random(self.seed + 9999)
                start = rng.randint(0, T - self.tdim)
            else:
                start = (T - self.tdim) // 2
            fb = fb[start:start + self.tdim]

        fb = (fb + NORM_ADD) / NORM_DIV
        x = fb.transpose(0, 1).unsqueeze(0)
        return x

    def __getitem__(self, idx):
        path, genre = self.items[idx]
        y, sr0 = self._read_audio(path)

        if sr0 != self.sr:
            r = self._resamplers.get(sr0)
            if r is None:
                r = Resample(sr0, self.sr)
                self._resamplers[sr0] = r
            y = r(y)

        y = self._crop(y, idx)
        x = self._to_fbank(y)
        label = self.label2idx[genre]
        return x, label

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

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def run_epoch(model, loader, criterion, optimizer, device, train: bool, use_amp: bool):
    model.train() if train else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), yb.detach()) * bs
        n += bs

    return total_loss / n, total_acc / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, type=str)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--crop_sec", type=float, default=10.0)
    ap.add_argument("--tdim", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--save_path", type=str, default=str(Path("..") / "artifacts" / "cnn_fbank_best.pt"))
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    splits_dir = Path(args.splits_dir).resolve()
    label2idx = json.loads((splits_dir / "label2idx.json").read_text(encoding="utf-8"))

    train_items = load_csv(splits_dir / "train.csv")
    val_items   = load_csv(splits_dir / "val.csv")
    test_items  = load_csv(splits_dir / "test.csv")

    train_ds = GTZANFbankDataset(train_items, label2idx, train=True,  sr=args.sr, crop_sec=args.crop_sec, tdim=args.tdim, seed=args.seed)
    val_ds   = GTZANFbankDataset(val_items,   label2idx, train=False, sr=args.sr, crop_sec=args.crop_sec, tdim=args.tdim, seed=args.seed)
    test_ds  = GTZANFbankDataset(test_items,  label2idx, train=False, sr=args.sr, crop_sec=args.crop_sec, tdim=args.tdim, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=args.pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=args.pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=args.pin_memory)

    device = torch.device(args.device)
    model = SmallCNN(n_classes=len(label2idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    save_path = Path(args.save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True,  use_amp=args.use_amp)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False, use_amp=args.use_amp)

        print(f"epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model_state": model.state_dict(), "label2idx": label2idx}, save_path)
            print("saved best to:", save_path)

    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False, use_amp=args.use_amp)
    print(f"TEST | loss {te_loss:.4f} acc {te_acc:.4f}")

if __name__ == "__main__":
    main()
