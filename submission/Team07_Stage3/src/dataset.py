"""Adapt the supplied labels_yolo layout without changing the source dataset."""
import shutil
from pathlib import Path

import yaml

from model import ROOT, read_config


def prepare_dataset(data_root):
    source = Path(data_root).resolve() / "detection"
    target = ROOT / "experiments" / "prepared_data"
    for source_split, target_split in (("train", "train"), ("validation", "val")):
        image_target = target / "images" / target_split
        label_target = target / "labels" / target_split
        image_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)
        for image in sorted((source / source_split / "images").glob("*.png")):
            image_copy = image_target / image.name
            label = source / source_split / "labels_yolo" / f"{image.stem}.txt"
            if not image_copy.exists():
                shutil.copy2(image, image_copy)
            if not (label_target / label.name).exists():
                shutil.copy2(label, label_target / label.name)
    # With no explicit path, Ultralytics resolves these relative to this YAML.
    specification = {
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(read_config()["class_order"])),
    }
    yaml_path = target / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(specification, sort_keys=False), encoding="utf-8")
    return yaml_path


def read_ground_truth(label_path, width, height):
    objects = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, xc, yc, box_width, box_height = map(float, line.split())
        objects.append({
            "class_id": int(class_id),
            "bbox": [(xc - box_width / 2) * width, (yc - box_height / 2) * height,
                     (xc + box_width / 2) * width, (yc + box_height / 2) * height],
        })
    return objects
