"""YOLO11n construction and loading for FactoryVision Stage 3."""
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def read_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def build_pretrained_model():
    """Load general COCO pretraining, never a defect-finetuned checkpoint."""
    checkpoint = ROOT / "weights" / "yolo11n.pt"
    return YOLO(str(checkpoint) if checkpoint.exists() else read_config()["pretrained_source"], task="detect")


def load_final_model():
    config = read_config()
    return YOLO(str(ROOT / config["checkpoint_file"]), task="detect")
