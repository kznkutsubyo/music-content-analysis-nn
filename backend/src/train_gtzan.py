from __future__ import annotations

from pathlib import Path
import json
import random
import csv

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.transforms import Resample

from models import ASTModel

NORM_ADD = 4.26
NORM_DIV = 4.57 * 2.0


def load_csv(csv_path: Path):
    items = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append((row["path"], row["genre"]))
    return items


class GTZANDataset(Dataset):
    def __init__(
        self,
        items,
        label2idx,
        train: bool,
        sr: int = 16000,
        crop_sec: float = 10.0,
        tdim: int = 1024,
        seed: int = 42,
    ):
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

    def _to_fbank(self, wav: torch.Tensor, sr: int):
        """
        wav: torch.Tensor [1, T] float32 на CPU
        return: [tdim, 128]
        """
        fb = torchaudio.compliance.kaldi.fbank(
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
        if T < self.tdim:
            fb = torch.nn.functional.pad(fb, (0, 0, 0, self.tdim - T))
        elif T > self.tdim:
            if self.train:
                start = random.randint(0, T - self.tdim)
            else:
                start = (T - self.tdim) // 2
            fb = fb[start : start + self.tdim]

        fb = (fb + NORM_ADD) / NORM_DIV
        return fb

    def _read_audio_soundfile(self, path: str):
        """
        Чтение аудио через soundfile без TorchCodec/FFmpeg.
        Возвращает torch.Tensor [1, T] float32 и sr (int)
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        audio, sr0 = sf.read(str(p), dtype="float32", always_2d=True)
        if audio.size == 0:
            raise RuntimeError(f"Empty audio file: {p}")

        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)

        y = torch.from_numpy(audio.T)
        return y, sr0

    def __getitem__(self, idx):
        path, genre = self.items[idx]

        y, sr0 = self._read_audio_soundfile(path)

        if sr0 != self.sr:
            resampler = self._resamplers.get(sr0)
            if resampler is None:
                resampler = Resample(sr0, self.sr)
                self._resamplers[sr0] = resampler
            y = resampler(y)

        crop_len = int(self.sr * self.crop_sec)
        if y.shape[1] < crop_len:
            y = torch.nn.functional.pad(y, (0, crop_len - y.shape[1]))
        elif y.shape[1] > crop_len:
            if self.train:
                start = random.randint(0, y.shape[1] - crop_len)
            else:
                start = (y.shape[1] - crop_len) // 2
            y = y[:, start : start + crop_len]

        fb = self._to_fbank(y, self.sr)

        label = self.label2idx[genre]
        return fb, label


def accuracy(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool, accum_steps: int = 1):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                (loss / accum_steps).backward()

                if step % accum_steps == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), yb.detach()) * bs
        n += bs

    return total_loss / n, total_acc / n


def save_checkpoint(path: Path, model, optimizer, label2idx, epoch: int, best_val: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "label2idx": label2idx,
            "epoch": epoch,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    for k in ["model_state", "label2idx"]:
        if k not in ckpt:
            raise KeyError(f"Checkpoint missing key: {k}")
    return ckpt


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop_sec", type=float, default=10.0)
    ap.add_argument("--tdim", type=int, default=1024)
    ap.add_argument("--model_size", type=str, default="base384")
    ap.add_argument("--accum_steps", type=int, default=1)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_path", type=str, default=str(Path("..") / "gtzan_ast_best.pt"))

    ap.add_argument("--resume", type=str, default="", help="Путь к чекпоинту для продолжения обучения")

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Ты указал --device cuda, но torch.cuda.is_available() == False.\n"
            "Проверь: 1) NVIDIA драйвер 2) что установлен CUDA-вариант torch (cu118/cu121), а не cpu."
        )

    device = torch.device(args.device)

    splits_dir = Path(args.splits_dir).resolve()

    label2idx = json.loads((splits_dir / "label2idx.json").read_text(encoding="utf-8"))

    train_items = load_csv(splits_dir / "train.csv")
    val_items = load_csv(splits_dir / "val.csv")
    test_items = load_csv(splits_dir / "test.csv")

    train_ds = GTZANDataset(train_items, label2idx, train=True, crop_sec=args.crop_sec, tdim=args.tdim, seed=args.seed)
    val_ds = GTZANDataset(val_items, label2idx, train=False, crop_sec=args.crop_sec, tdim=args.tdim, seed=args.seed)
    test_ds = GTZANDataset(test_items, label2idx, train=False, crop_sec=args.crop_sec, tdim=args.tdim, seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = ASTModel(
        label_dim=len(label2idx),
        input_tdim=args.tdim,
        imagenet_pretrain=True,
        audioset_pretrain=False,
        model_size=args.model_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_path = Path(args.save_path).resolve()

    start_epoch = 1
    best_val = -1.0

    if args.resume:
        resume_path = Path(args.resume).resolve()
        ckpt = load_checkpoint(resume_path, device)

        model.load_state_dict(ckpt["model_state"], strict=True)

        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", -1.0))

        label2idx = ckpt.get("label2idx", label2idx)

        print(f"RESUME: loaded {resume_path}")
        print(f"RESUME: start_epoch={start_epoch}, best_val={best_val}")

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True,
                                    accum_steps=args.accum_steps)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False, accum_steps=1)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}",
            flush=True,
        )

        if va_acc > best_val:
            best_val = va_acc
            save_checkpoint(save_path, model, optimizer, label2idx, epoch, best_val)
            print("saved best to:", save_path, flush=True)

    ckpt = load_checkpoint(save_path, device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    te_loss, te_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"TEST | loss {te_loss:.4f} acc {te_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
