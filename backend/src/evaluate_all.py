from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.transforms import Resample

from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from joblib import load as joblib_load

import matplotlib.pyplot as plt

from models import ASTModel


NORM_ADD = 4.26
NORM_DIV = 4.57 * 2.0


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_split_csv(csv_path: Path):
    items = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append((row["path"], row["genre"]))
    return items


def resolve_audio_path(path_str: str, splits_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    cand1 = (splits_dir / p).resolve()
    if cand1.exists():
        return cand1
    cand2 = (splits_dir.parent / p).resolve()
    if cand2.exists():
        return cand2
    return p.resolve()


def read_audio_soundfile(path: Path):
    audio, sr0 = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise RuntimeError(f"Empty audio file: {path}")
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
        rng = np.random.default_rng(seed + idx)
        start = int(rng.integers(0, T - crop_len + 1))
    else:
        start = (T - crop_len) // 2
    return y[:, start : start + crop_len]


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45, ha="right")
    plt.yticks(tick, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_eval_artifacts(out_dir: Path, y_true, y_pred, proba, labels: list[str], method_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = float(accuracy_score(y_true, y_pred))
    top3 = float(top_k_accuracy_score(y_true, proba, k=min(3, proba.shape[1])))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    write_json(out_dir / "metrics.json", {
        "accuracy": acc,
        "top3_accuracy": top3,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.save(out_dir / "confusion_matrix.npy", cm)
    plot_confusion_matrix(cm, labels, f"Confusion Matrix ({method_name})", out_dir / "confusion_matrix.png")

    return {
        "accuracy": acc,
        "top3_accuracy": top3,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


class FbankDataset(Dataset):
    def __init__(self, items, label2idx, splits_dir: Path,
                 sr: int = 16000, crop_sec: float = 10.0, tdim: int = 1024,
                 train: bool = False, seed: int = 42):
        self.items = items
        self.label2idx = label2idx
        self.splits_dir = splits_dir
        self.sr = sr
        self.crop_sec = crop_sec
        self.tdim = tdim
        self.train = train
        self.seed = seed
        self._resamplers: dict[int, Resample] = {}

    def __len__(self):
        return len(self.items)

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
            start = (T - self.tdim) // 2
            fb = fb[start : start + self.tdim]

        fb = (fb + NORM_ADD) / NORM_DIV
        return fb

    def __getitem__(self, idx):
        path_str, genre = self.items[idx]
        path = resolve_audio_path(path_str, self.splits_dir)

        y, sr0 = read_audio_soundfile(path)

        if sr0 != self.sr:
            r = self._resamplers.get(sr0)
            if r is None:
                r = Resample(sr0, self.sr)
                self._resamplers[sr0] = r
            y = r(y)

        y = crop_or_pad(y, self.sr, self.crop_sec, train=self.train, seed=self.seed, idx=idx)
        fb = self._to_fbank(y)
        label = self.label2idx[genre]
        return fb, label


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


def eval_torch_model(model, loader, device: torch.device, amp: bool = True):
    model.eval()
    y_true = []
    proba_all = []

    t0 = time.perf_counter()
    n_items = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                logits = model(xb)

            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            proba_all.append(prob)
            y_true.append(yb.detach().cpu().numpy())
            n_items += xb.size(0)

    dt = time.perf_counter() - t0
    y_true = np.concatenate(y_true, axis=0)
    proba = np.concatenate(proba_all, axis=0)
    y_pred = np.argmax(proba, axis=1)

    ms_per_item = 1000.0 * dt / max(1, n_items)
    return y_true, y_pred, proba, ms_per_item


def mfcc_stats_from_path(path: Path, target_sr: int, crop_sec: float):
    y, sr0 = read_audio_soundfile(path)

    if sr0 != target_sr:
        y = Resample(sr0, target_sr)(y)

    y = crop_or_pad(y, target_sr, crop_sec, train=False, seed=0, idx=0)

    mfcc = torchaudio.compliance.kaldi.mfcc(
        y,
        sample_frequency=target_sr,
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


def try_load_mfcc_npz(features_dir: Path):
    tr = features_dir / "train_mfcc.npz"
    va = features_dir / "val_mfcc.npz"
    te = features_dir / "test_mfcc.npz"
    if te.exists():
        d = np.load(te, allow_pickle=True)
        return d["X"], d["y"], d.get("paths", None)
    return None


def main():
    ROOT = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()

    ap.add_argument("--splits_dir", required=True, type=str)
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "results_compare"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--ast_ckpt", type=str, default=str(ROOT / "gtzan_ast_best.pt"))
    ap.add_argument("--ast_model_size", type=str, default="base384")
    ap.add_argument("--tdim", type=int, default=1024)
    ap.add_argument("--crop_sec", type=float, default=10.0)

    ap.add_argument("--cnn_ckpt", type=str, default=str(ROOT / "artifacts" / "cnn_fbank_best.pt"))

    ap.add_argument("--features_dir", type=str, default=str(ROOT.parent / "features_mfcc"))
    ap.add_argument("--knn_model", type=str, default=str(ROOT / "artifacts" / "knn_mfcc.joblib"))
    ap.add_argument("--rf_model", type=str, default=str(ROOT / "artifacts" / "rf_mfcc.joblib"))

    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    amp = not args.no_amp

    label2idx = read_json(splits_dir / "label2idx.json")
    idx2label = {int(v): k for k, v in label2idx.items()}
    labels = [idx2label[i] for i in range(len(idx2label))]

    test_items = load_split_csv(splits_dir / "test.csv")

    summary_rows = []

    ast_ckpt = Path(args.ast_ckpt).resolve()
    if ast_ckpt.exists():
        print("EVAL AST:", ast_ckpt)

        ckpt = torch.load(ast_ckpt, map_location=device)
        ckpt_label2idx = ckpt.get("label2idx", None)
        if ckpt_label2idx is not None and ckpt_label2idx != label2idx:
            print("WARNING: label2idx in checkpoint differs from splits_dir label2idx. Using splits_dir label2idx.")

        ds = FbankDataset(
            test_items, label2idx, splits_dir,
            sr=16000, crop_sec=args.crop_sec, tdim=args.tdim,
            train=False, seed=42
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=args.pin_memory)

        model = ASTModel(
            label_dim=len(label2idx),
            input_tdim=args.tdim,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size=args.ast_model_size,
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        y_true, y_pred, proba, ms_item = eval_torch_model(model, loader, device, amp=amp)
        metrics = save_eval_artifacts(out_dir / "ast", y_true, y_pred, proba, labels, "AST")
        metrics["ms_per_item"] = float(ms_item)
        write_json(out_dir / "ast" / "speed.json", {"ms_per_item": float(ms_item)})

        summary_rows.append({
            "method": "AST",
            **metrics,
        })
    else:
        print("SKIP AST: checkpoint not found:", ast_ckpt)

    cnn_ckpt = Path(args.cnn_ckpt).resolve()
    if cnn_ckpt.exists():
        print("EVAL CNN:", cnn_ckpt)

        ckpt = torch.load(cnn_ckpt, map_location=device)
        ds = FbankDataset(
            test_items, label2idx, splits_dir,
            sr=16000, crop_sec=args.crop_sec, tdim=args.tdim,
            train=False, seed=42
        )

        class WrapCNN(Dataset):
            def __init__(self, base: Dataset):
                self.base = base
            def __len__(self):
                return len(self.base)
            def __getitem__(self, i):
                fb, y = self.base[i]
                x = fb.transpose(0, 1).unsqueeze(0)
                return x, y

        loader = DataLoader(WrapCNN(ds), batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=args.pin_memory)

        model = SmallCNN(n_classes=len(label2idx)).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        y_true, y_pred, proba, ms_item = eval_torch_model(model, loader, device, amp=amp)
        metrics = save_eval_artifacts(out_dir / "cnn", y_true, y_pred, proba, labels, "CNN")
        metrics["ms_per_item"] = float(ms_item)
        write_json(out_dir / "cnn" / "speed.json", {"ms_per_item": float(ms_item)})

        summary_rows.append({
            "method": "CNN",
            **metrics,
        })
    else:
        print("SKIP CNN: checkpoint not found:", cnn_ckpt)

    features_dir = Path(args.features_dir).resolve()
    test_npz = features_dir / "test_mfcc.npz"

    Xte = yte = None
    if test_npz.exists():
        d = np.load(test_npz, allow_pickle=True)
        Xte = d["X"]
        yte = d["y"]
    else:
        print("NOTE: MFCC features not found:", test_npz)
        print("      You can generate them with:")
        print(f"      python src\\extract_mfcc_features.py --splits_dir {splits_dir} --out_dir {features_dir} --crop_sec 30")

    def eval_sklearn(name: str, model_path: Path, out_subdir: str):
        nonlocal Xte, yte
        if not model_path.exists():
            print(f"SKIP {name}: model not found:", model_path)
            return

        if Xte is None or yte is None:
            print(f"{name}: computing MFCC on the fly (slow)...")
            feats = []
            labels_y = []
            for i, (pstr, genre) in enumerate(test_items):
                p = resolve_audio_path(pstr, splits_dir)
                feat = mfcc_stats_from_path(p, target_sr=16000, crop_sec=30.0)
                feats.append(feat)
                labels_y.append(label2idx[genre])
            Xte = np.stack(feats, axis=0)
            yte = np.array(labels_y, dtype=np.int64)

        clf = joblib_load(model_path)
        t0 = time.perf_counter()
        proba = clf.predict_proba(Xte)
        dt = time.perf_counter() - t0
        y_pred = np.argmax(proba, axis=1)
        ms_item = 1000.0 * dt / max(1, len(yte))

        metrics = save_eval_artifacts(out_dir / out_subdir, yte, y_pred, proba, labels, name)
        metrics["ms_per_item"] = float(ms_item)
        write_json(out_dir / out_subdir / "speed.json", {"ms_per_item": float(ms_item)})

        summary_rows.append({
            "method": name,
            **metrics,
        })

    eval_sklearn("KNN", Path(args.knn_model).resolve(), "knn")
    eval_sklearn("RandomForest", Path(args.rf_model).resolve(), "rf")

    if summary_rows:
        order = {"AST": 0, "CNN": 1, "RandomForest": 2, "KNN": 3}
        summary_rows.sort(key=lambda r: order.get(r["method"], 999))

        summary_csv = out_dir / "summary_table.csv"
        cols = ["method", "accuracy", "top3_accuracy", "macro_f1", "weighted_f1", "ms_per_item"]
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in summary_rows:
                w.writerow({c: row.get(c, "") for c in cols})

        write_json(out_dir / "summary_table.json", summary_rows)

        print("\n=== SUMMARY ===")
        for r in summary_rows:
            print(
                f"{r['method']:<12} acc={r['accuracy']:.4f} top3={r['top3_accuracy']:.4f} "
                f"macroF1={r['macro_f1']:.4f} ms/item={r.get('ms_per_item', 0):.2f}"
            )
        print("Saved summary to:", summary_csv)
        print("All outputs in:", out_dir)
    else:
        print("Nothing evaluated (no models found).")


if __name__ == "__main__":
    main()
