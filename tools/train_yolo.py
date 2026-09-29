"""Training & fine-tuning script for Clash Royale YOLOv8 detector.

Supports:
1. Local training using Ultralytics YOLOv8 on Clash-royale-1/data.yaml.
2. Generating a Kaggle training notebook / submission script using kaggleapikey.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT_DIR / "Clash-royale-1" / "data.yaml"
BASE_MODEL = ROOT_DIR / "yolov8n.pt"
OUTPUT_MODEL = ROOT_DIR / "clash_royale_yolo.pt"


def train_local(
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "cpu",
    resume_model: Path | None = None,
) -> Path:
    """Train or fine-tune YOLOv8 on the Clash-royale-1 dataset."""
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration not found at {DATASET_YAML}")

    model_source = resume_model if (resume_model and resume_model.is_file()) else BASE_MODEL
    print(f"Loading base model from {model_source}...")
    model = YOLO(str(model_source))

    print(f"Starting training: {epochs} epochs, batch {batch_size}, img size {imgsz}, device {device}...")
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(ROOT_DIR / "runs" / "train"),
        name="cr_yolo",
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.is_file():
        print(f"Training completed successfully! Best weights: {best_weights}")
        import shutil

        shutil.copy2(best_weights, OUTPUT_MODEL)
        print(f"Updated active bot model: {OUTPUT_MODEL}")
        return OUTPUT_MODEL

    return Path(results.save_dir)


def generate_kaggle_script() -> Path:
    """Generate a Kaggle Notebook / CLI script to run training on Kaggle GPU."""
    kaggle_key_file = Path(r"C:\Users\Gökçen\Desktop\kaggleapikey.txt")
    kaggle_key = kaggle_key_file.read_text().strip() if kaggle_key_file.is_file() else ""

    script_content = f"""# Kaggle GPU Training Script for Clash Royale YOLOv8
# Kaggle API Key: {kaggle_key}

import os
import shutil
from ultralytics import YOLO

# Install dependencies
# !pip install ultralytics

# Clone dataset / load dataset
# model = YOLO("yolov8n.pt")
# model.train(data="data.yaml", epochs=100, imgsz=640, device=0, batch=32)
# model.export(format="onnx")
"""
    out_script = ROOT_DIR / "tools" / "kaggle_train_guide.py"
    out_script.write_text(script_content, encoding="utf-8")
    return out_script


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Clash Royale YOLO detector")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_local(epochs=args.epochs, batch_size=args.batch, device=args.device)
