#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import shutil
import yaml
from ultralytics import YOLO


def check_dataset_structure(dataset_path: Path):
    expected = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
        "test/images", "test/labels",
        "data.yaml"
    ]
    missing = []
    for e in expected:
        p = dataset_path / e
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: Struktur dataset tidak lengkap. Missing:")
        for m in missing:
            print(" -", m)
        sys.exit(1)
    print("Dataset structure OK.")


def load_data_yaml(path: Path):
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    print("Loaded data.yaml:", path)
    return d


def find_best_weights(run_dir: Path):
    weights_dir = run_dir / "weights"
    if not weights_dir.exists():
        return None
    candidates = list(weights_dir.glob("*.pt"))
    if not candidates:
        return None
    # choose best.pt if exists else largest file
    for c in candidates:
        if c.name.lower() == "best.pt" or c.name.lower().startswith("best"):
            return c
    return max(candidates, key=lambda p: p.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to vehicle-detection.v1i.yolov12")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="pretrained model or backbone")
    parser.add_argument("--output", type=str, default="models")
    parser.add_argument("--export", action="store_true", help="Export to onnx and torchscript after training")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print("ERROR: dataset_path tidak ditemukan:", dataset_path)
        sys.exit(1)

    check_dataset_structure(dataset_path)
    data_yaml = dataset_path / "data.yaml"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model:", args.model)
    model = YOLO(args.model)

    print("Start training. This may take time depending on GPU.")
    # train
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        save=True,
        project=str(output_dir),
        name="yolov8_training",
        exist_ok=True
    )

    run_dir = output_dir / "yolov8_training"
    best = find_best_weights(run_dir)
    if best:
        final_pt = output_dir / "model_best.pt"
        try:
            shutil.copy(best, final_pt)
            print("Copied best weights to:", final_pt)
        except Exception as e:
            print("Gagal menyalin weights:", e)
        if args.export:
            print("Exporting model (ONNX / TorchScript)... This uses ultralytics export API.")
            try:
                exp = YOLO(str(final_pt))
                exp.export(format="onnx", imgsz=args.img)
                exp.export(format="torchscript", imgsz=args.img)
                print("Export completed (check current working dir for exported files).")
            except Exception as e:
                print("Export failed:", e)
    else:
        print("Tidak menemukan weights hasil training di:", run_dir)


if __name__ == "__main__":
    main()
