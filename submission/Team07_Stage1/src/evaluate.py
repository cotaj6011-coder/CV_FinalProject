"""Evaluate the final saved CNN without fitting on validation data."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (DefectDataset, evaluate_model, measure_inference,
                    save_confusion, save_failures, write_json)
from model import load_checkpoint


def main():
    stage_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=stage_root / "config.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=stage_root / "evaluation")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint_path = args.checkpoint or stage_root / config["checkpoint_file"]
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    for key in ("input_size", "class_order", "normalization_mean", "normalization_std", "dropout"):
        if config[key] != checkpoint[key]:
            raise ValueError(f"Config and checkpoint disagree: {key}")
    dataset = DefectDataset(args.data_root, "validation", config)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])
    metrics, records, paths = evaluate_model(model, loader, device, config["class_order"])
    metrics["inference"] = measure_inference(model, dataset, device)
    metrics["checkpoint_file"] = checkpoint_path.name
    metrics["selected_experiment"] = checkpoint["experiment"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "predictions.json", records)
    save_confusion(metrics["confusion_matrix"], config["class_order"],
                   args.output_dir / "confusion_matrix.png", "Stage 1 final CNN")
    failures = save_failures(records, paths, config["class_order"], args.output_dir / "failure_cases.png")
    write_json(args.output_dir / "failure_cases.json", failures)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
