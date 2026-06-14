from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score

from train_gtzan import GTZANDataset, load_csv
from models import ASTModel

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(Path("../..") / "gtzan_ast_best.pt"))
    ap.add_argument("--splits_dir", type=str, default=str(Path("../..") / "gtzan_splits"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tdim", type=int, default=512)
    ap.add_argument("--crop_sec", type=float, default=5.0)
    ap.add_argument("--model_size", type=str, default="small224")  # вместо base384
    ap.add_argument("--accum_steps", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    label2idx = ckpt["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}
    labels_sorted = [idx2label[i] for i in range(len(idx2label))]

    splits_dir = Path(args.splits_dir).resolve()
    test_items = load_csv(splits_dir / "test.csv")
    test_ds = GTZANDataset(test_items, label2idx, train=False, crop_sec=args.crop_sec, tdim=args.tdim)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(args.device.startswith("cuda")))

    model = ASTModel(
        label_dim=len(label2idx),
        input_tdim=args.tdim,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size=args.model_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob.append(prob)
            y_true.append(yb.numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    top3 = top_k_accuracy_score(y_true, y_prob, k=3, labels=list(range(len(label2idx))))

    print("TEST accuracy:", acc)
    print("TEST top-3 accuracy:", top3)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=labels_sorted, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    out = Path("../..") / "results"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "confusion_matrix.npy", cm)
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=labels_sorted, digits=4),
        encoding="utf-8"
    )
    print("\nSaved to:", out.resolve())

if __name__ == "__main__":
    main()
