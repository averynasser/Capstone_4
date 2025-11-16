
import argparse
from pathlib import Path
import shutil
import yaml
import json
import pandas as pd
from ultralytics import YOLO


def load_data_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset root (contains data.yaml)")
    parser.add_argument("--output", default="reports", help="Output folder for reports")
    parser.add_argument("--conf", type=float, default=0.001, help="Minimum conf for evaluation stage")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    ds = Path(args.dataset_path)
    data_yaml = ds / "data.yaml"
    if not data_yaml.exists():
        print("data.yaml tidak ditemukan pada dataset_path:", ds)
        return

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading model:", args.model)
    model = YOLO(args.model)

    print("Running COCO-style evaluation (this may take a while)...")
    results = model.val(
        data=str(data_yaml),
        split="test",
        conf=args.conf,
        iou=args.iou,
        save_json=True,
        plots=True,
        project=str(out),
        name="val_results",
        exist_ok=True,
    )

    # results contains structured metrics; extract useful fields
    try:
        metrics_summary = {
            "mAP_50": float(results.box.map50),
            "mAP_50_95": float(results.box.map),
            "precision_mean": float(results.box.p.mean()),
            "recall_mean": float(results.box.r.mean()),
        }
    except Exception:
        metrics_summary = {"error": "failed to read metrics from results object"}

    print("Metrics summary:", metrics_summary)

    # save summary & per-class csv if available
    try:
        with open(out / "evaluation_report.json", "w") as f:
            f.write(json.dumps(metrics_summary, indent=2))
    except Exception as e:
        print("Gagal menyimpan evaluation_report.json:", e)

    try:
        names = results.names  # mapping id->name
        df = pd.DataFrame({
            "class_id": list(names.keys()),
            "class_name": list(names.values()),
            "precision": results.box.p.tolist() if hasattr(results.box.p, "tolist") else list(results.box.p),
            "recall": results.box.r.tolist() if hasattr(results.box.r, "tolist") else list(results.box.r),
            "mAP@0.5": results.box.map50_per_class.tolist(),
            "mAP@0.5:0.95": results.box.map_per_class.tolist(),
        })
        df.to_csv(out / "evaluation_per_class.csv", index=False)
    except Exception as e:
        print("Gagal menyimpan per-class csv:", e)

    print("Saved evaluation outputs to", out)


if __name__ == "__main__":
    main()
