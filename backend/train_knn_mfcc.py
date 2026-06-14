from __future__ import annotations
from pathlib import Path
import argparse
import json

import numpy as np
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_npz(p: Path):
    d = np.load(p, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    paths = d["paths"] if "paths" in d.files else None
    return X, y, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, required=True, help="folder with train_mfcc.npz/val_mfcc.npz/test_mfcc.npz")
    ap.add_argument("--save_path", type=str, default=str(Path("artifacts") / "knn_mfcc.joblib"))
    ap.add_argument("--k_list", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11, 15, 21])
    ap.add_argument("--weights", type=str, default="distance", choices=["uniform", "distance"])
    args = ap.parse_args()

    features_dir = Path(args.features_dir).resolve()
    tr_p = features_dir / "train_mfcc.npz"
    va_p = features_dir / "val_mfcc.npz"
    te_p = features_dir / "test_mfcc.npz"

    if not tr_p.exists() or not va_p.exists() or not te_p.exists():
        raise FileNotFoundError(
            "Не нашёл нужные файлы признаков. Ожидаю:\n"
            f"  {tr_p}\n  {va_p}\n  {te_p}\n"
            "Сначала запусти extract_mfcc_features.py"
        )

    Xtr, ytr, _ = load_npz(tr_p)
    Xva, yva, _ = load_npz(va_p)
    Xte, yte, _ = load_npz(te_p)

    best_k = None
    best_acc = -1.0
    best_model = None

    for k in args.k_list:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights=args.weights)),
        ])
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        acc = accuracy_score(yva, pred)

        print(f"k={k:>2} | val acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = model

    print(f"\nBEST: k={best_k}, val acc={best_acc:.4f}")

    te_pred = best_model.predict(Xte)
    te_acc = accuracy_score(yte, te_pred)
    print(f"TEST acc={te_acc:.4f}")

    save_path = Path(args.save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dump(best_model, save_path)
    print("Saved KNN model to:", save_path)

    report = classification_report(yte, te_pred, digits=4)
    (save_path.parent / "knn_test_report.txt").write_text(report, encoding="utf-8")
    np.save(save_path.parent / "knn_test_confusion_matrix.npy", confusion_matrix(yte, te_pred))


if __name__ == "__main__":
    main()
