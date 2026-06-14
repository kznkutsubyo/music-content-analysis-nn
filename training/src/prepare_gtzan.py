# prepare_gtzan.py
from __future__ import annotations
from pathlib import Path
import json
import random
import csv

def find_audio_root(data_root: Path) -> Path:
    candidates = [
        data_root / "genres_original",
        data_root / "genres",
        data_root,
    ]
    for c in candidates:
        if c.exists() and any(c.rglob("*.wav")):
            return c
    raise FileNotFoundError(f"Не нашёл .wav внутри {data_root}. Проверь структуру датасета.")

def collect_files(audio_root: Path):
    rows = []
    for genre_dir in sorted([p for p in audio_root.iterdir() if p.is_dir()]):
        genre = genre_dir.name
        for wav in sorted(genre_dir.glob("*.wav")):
            rows.append((str(wav.resolve()), genre))
    if not rows:
        # fallback: если вдруг wav лежат глубже
        for wav in sorted(audio_root.rglob("*.wav")):
            # пробуем взять жанр как имя родительской папки
            genre = wav.parent.name
            rows.append((str(wav.resolve()), genre))
    return rows

def stratified_split(rows, seed=42, train=0.8, val=0.1, test=0.1):
    random.seed(seed)
    by_genre = {}
    for path, genre in rows:
        by_genre.setdefault(genre, []).append(path)
    for g in by_genre:
        random.shuffle(by_genre[g])

    train_rows, val_rows, test_rows = [], [], []
    for genre, paths in by_genre.items():
        n = len(paths)
        n_train = int(n * train)
        n_val = int(n * val)
        n_test = n - n_train - n_val
        train_rows += [(p, genre) for p in paths[:n_train]]
        val_rows   += [(p, genre) for p in paths[n_train:n_train+n_val]]
        test_rows  += [(p, genre) for p in paths[n_train+n_val:]]
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)
    return train_rows, val_rows, test_rows

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "genre"])
        w.writerows(rows)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Папка, куда распакован Kaggle датасет")
    ap.add_argument("--out_dir", type=str, default=str(Path("..") / "gtzan_splits"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    audio_root = find_audio_root(data_root)
    rows = collect_files(audio_root)

    genres = sorted({g for _, g in rows})
    label2idx = {g: i for i, g in enumerate(genres)}
    (out_dir).mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, test_rows = stratified_split(rows, seed=args.seed)

    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)
    write_csv(out_dir / "test.csv", test_rows)

    with (out_dir / "label2idx.json").open("w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)

    print("OK")
    print("audio_root:", audio_root)
    print("genres:", genres)
    print("train/val/test:", len(train_rows), len(val_rows), len(test_rows))
    print("out_dir:", out_dir)

if __name__ == "__main__":
    main()
