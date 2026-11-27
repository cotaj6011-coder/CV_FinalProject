"""Train YOLO11n and select the checkpoint by validation mAP@0.5."""
import argparse
import csv
import json
import os

from dataset import prepare_dataset
from export import export_inference_checkpoint
from model import ROOT, build_pretrained_model, read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to STUDENT_PUBLIC")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    args = parser.parse_args()
    data_yaml = prepare_dataset(args.data_root)
    config = read_config()
    os.chdir(ROOT)
    model = build_pretrained_model()
    model.train(
        data=str(data_yaml), imgsz=config["input_size"], seed=config["seed"],
        device=args.device, project=str(ROOT / "experiments"), name="train",
        exist_ok=False, save=True, save_period=1, plots=True, val=True,
        **config["training"],
    )
    run_directory = model.trainer.save_dir
    with (run_directory / "results.csv").open(encoding="utf-8") as handle:
        rows = [{key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(handle)]
    # Ultralytics rows number epochs from 1; saved epoch filenames start at 0.
    best = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    epoch = int(best["epoch"])
    selected = run_directory / "weights" / f"epoch{epoch - 1}.pt"
    export_inference_checkpoint(selected, ROOT / config["checkpoint_file"])
    selection = {
        "selection_metric": config["selection_metric"],
        "selected_epoch": epoch,
        "selected_validation_map50": float(best["metrics/mAP50(B)"]),
        "source_checkpoint": str(selected.relative_to(ROOT)).replace("\\", "/"),
        "epochs_completed": len(rows),
        "training_results_csv": str((run_directory / "results.csv").relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_file": config["checkpoint_file"],
    }
    (ROOT / "experiments" / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
