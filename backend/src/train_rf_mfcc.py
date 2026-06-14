# train_rf_mfcc.py
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np

from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

def load_npz(p: Path):
    d = np.load(p, allow_pickle=True)
    return d["X"], d["y"], d["paths"]

def save_results(out_dir: Path, y_true, y_pred, proba, idx2label):
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    top3 = top_k_accuracy_score(y_true, proba, k=3)

    report = classification_report(
        y_true, y_pred,
        target_names=[idx2label[str(i)] for i in range(len(idx2label))],
        digits=4
    )
    cm = confusion_matrix(y_true, y_pred)

    (out_dir / "metrics.json").write_text(json.dumps({"accuracy": float(acc), "top3": float(top3)}, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.save(out_dir / "confusion_matrix.npy", cm)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (RF MFCC)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print("TEST accuracy:", acc)
    print("TEST top-3 accuracy:", top3)
    print("\nClassification report:\n", report)
    print("Saved to:", out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True, type=str)
    ap.add_argument("--out_model", default=str(Path("..") / "artifacts" / "rf_mfcc.joblib"))
    ap.add_argument("--out_results", default=str(Path("..") / "results" / "rf"))
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=0)  # 0 => None
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    features_dir = Path(args.features_dir).resolve()
    Xtr, ytr, _ = load_npz(features_dir / "train_mfcc.npz")
    Xva, yva, _ = load_npz(features_dir / "val_mfcc.npz")
    Xte, yte, _ = load_npz(features_dir / "test_mfcc.npz")

    idx2label = json.loads((features_dir / "idx2label.json").read_text(encoding="utf-8"))

    # train on train+val
    X_all = np.concatenate([Xtr, Xva], axis=0)
    y_all = np.concatenate([ytr, yva], axis=0)

    max_depth = None if args.max_depth == 0 else args.max_depth

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        random_state=args.seed,
        n_jobs=-1
    )
    clf.fit(X_all, y_all)

    out_model = Path(args.out_model).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, out_model)
    print("Saved model:", out_model)

    proba = clf.predict_proba(Xte)
    y_pred = np.argmax(proba, axis=1)

    save_results(Path(args.out_results).resolve(), yte, y_pred, proba, idx2label)

if __name__ == "__main__":
    main()
