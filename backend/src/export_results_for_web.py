# src/export_results_for_web.py
from pathlib import Path
import json

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    root = Path(__file__).resolve().parents[1]   # ast/
    res = root / "results_compare"
    out = root / "web_results.json"

    methods = ["ast", "cnn", "rf", "knn"]
    payload = {"methods": []}

    for m in methods:
        d = res / m
        if not d.exists():
            continue
        metrics = read_json(d / "metrics.json") if (d / "metrics.json").exists() else {}
        speed = read_json(d / "speed.json") if (d / "speed.json").exists() else {}
        payload["methods"].append({
            "id": m,
            "name": {"ast": "AST", "cnn": "CNN", "rf": "RandomForest", "knn": "KNN"}[m],
            "metrics": metrics,
            "speed": speed,
            "artifacts": {
                "confusion_matrix_png": str((d / "confusion_matrix.png").as_posix()),
                "classification_report_txt": str((d / "classification_report.txt").as_posix()),
            }
        })

    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out)

if __name__ == "__main__":
    main()
