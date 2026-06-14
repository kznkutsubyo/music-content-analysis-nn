from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

def get_labels():
    # Пытаемся взять правильный порядок меток из label2idx.json
    splits = Path(__file__).resolve().parents[1] / "gtzan_splits" / "label2idx.json"  # ast/gtzan_splits/...
    if splits.exists():
        label2idx = json.loads(splits.read_text(encoding="utf-8"))
        idx2label = {v: k for k, v in label2idx.items()}
        return [idx2label[i] for i in range(len(idx2label))]
    # fallback (если файла нет)
    return ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def main():
    # results лежит в bc/results (на 2 уровня выше, чем src)
    results_dir = Path(__file__).resolve().parents[2] / "results"  # ...\bc\results
    cm_path = results_dir / "confusion_matrix.npy"

    if not cm_path.exists():
        raise FileNotFoundError(f"Не найден {cm_path}. Проверь, что evaluate сохранил results в bc\\results.")

    cm = np.load(cm_path)
    labels = get_labels()

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("GTZAN - Confusion Matrix (AST)")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    out = results_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=200)
    print("Saved:", out.resolve())

if __name__ == "__main__":
    main()
