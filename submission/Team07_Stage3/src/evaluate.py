"""Evaluate the selected detector, save evidence, and benchmark separately."""
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import ultralytics
from PIL import Image

from dataset import prepare_dataset, read_ground_truth
from metrics import evaluate_threshold, select_failure_cases
from model import ROOT, load_final_model, read_config
from plots import detection_pair, per_class_ap_plot, threshold_plot


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def environment(device):
    gpu = device != "cpu" and torch.cuda.is_available()
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(int(device)) if gpu else "CPU",
    }


def predict_validation(model, data_yaml, config, device):
    image_directory = data_yaml.parent / "images" / "val"
    label_directory = data_yaml.parent / "labels" / "val"
    records = []
    results = model.predict(
        source=str(image_directory), stream=True, batch=16,
        imgsz=config["input_size"], device=device, half=False, verbose=False,
        conf=config["evaluation"]["ap_min_confidence"],
        iou=config["nms_iou_threshold"], max_det=config["evaluation"]["max_detections"],
        save=False,
    )
    for result in results:
        height, width = result.orig_shape
        image_name = Path(result.path).name
        boxes = result.boxes
        predictions = [{"class_id": int(class_id), "confidence": float(confidence),
                        "bbox": [float(value) for value in coordinates]}
                       for coordinates, class_id, confidence in zip(
                           boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist())]
        records.append({
            "image": image_name, "width": width, "height": height, "bbox_format": "xyxy_pixel",
            "ground_truth": read_ground_truth(label_directory / f"{Path(image_name).stem}.txt", width, height),
            "predictions": predictions,
        })
    return sorted(records, key=lambda item: item["image"])


def save_visualizations(records, failures, data_yaml, config, output):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    selected = {}
    # Deterministic coverage: first two validation images containing each class.
    for class_id in range(len(config["class_order"])):
        candidates = [record for record in records
                      if any(gt["class_id"] == class_id for gt in record["ground_truth"])]
        for record in candidates[:2]:
            selected[record["image"]] = record
    for record in records:
        if len(selected) >= 12:
            break
        selected[record["image"]] = record
    manifest = []
    image_directory = data_yaml.parent / "images" / "val"
    for record in selected.values():
        destination = figures / f"comparison_{Path(record['image']).stem}.png"
        detection_pair(image_directory / record["image"], record, config["confidence_threshold"],
                       config["class_order"], destination)
        manifest.append(str(destination.relative_to(ROOT)).replace("\\", "/"))
    by_image = {record["image"]: record for record in records}
    for index, case in enumerate(failures, 1):
        destination = figures / f"failure_{index:02d}_{Path(case['image']).stem}.png"
        detection_pair(image_directory / case["image"], by_image[case["image"]],
                       config["confidence_threshold"], config["class_order"], destination)
        case["figure"] = str(destination.relative_to(ROOT)).replace("\\", "/")
    return manifest


def benchmark(model, data_yaml, config, device):
    image_paths = sorted((data_yaml.parent / "images" / "val").glob("*.png"))
    images = [np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy() for path in image_paths]
    arguments = dict(imgsz=config["input_size"], device=device, half=False,
                     conf=config["confidence_threshold"], iou=config["nms_iou_threshold"],
                     verbose=False, save=False, max_det=config["evaluation"]["max_detections"])
    gpu = device != "cpu" and torch.cuda.is_available()
    for index in range(20):
        model.predict(images[index % len(images)], **arguments)[0].boxes.data.cpu().tolist()
    timings, forward_timings = [], []
    for image in images:
        if gpu:
            torch.cuda.synchronize(int(device))
        started = time.perf_counter()
        result = model.predict(image, **arguments)[0]
        result.boxes.data.cpu().tolist()
        if gpu:
            torch.cuda.synchronize(int(device))
        timings.append((time.perf_counter() - started) * 1000)
        forward_timings.append(float(result.speed["inference"]))
    return {
        "protocol": "batch=1; FP32; 20 warmups; all 270 validation BGR images preloaded in memory; CUDA synchronized; excludes disk I/O and model loading; includes preprocessing, transfer, prediction API, model, NMS and output boxes returned to CPU",
        "samples": len(timings), "warmup_iterations": 20,
        "precision": "FP32", "batch_size": 1,
        "image_ids": [path.name for path in image_paths],
        "gt_class_image_counts": {str(class_id): sum(any(gt['class_id'] == class_id for gt in read_ground_truth(
            data_yaml.parent / 'labels' / 'val' / f'{path.stem}.txt', 200, 200)) for path in image_paths)
            for class_id in range(6)},
        "end_to_end_mean_ms": float(np.mean(timings)),
        "end_to_end_median_ms": float(np.median(timings)),
        "end_to_end_p95_ms": float(np.percentile(timings, 95)),
        "model_forward_mean_ms": float(np.mean(forward_timings)),
        "estimated_sequential_fps": 1000 / float(np.mean(timings)),
        "sample_ms": timings, "environment": environment(device),
    }


def run_benchmark(data_root, device):
    """Measure once under the recorded settings and refresh all latency evidence."""
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data_yaml = prepare_dataset(data_root)
    config = read_config()
    latency = benchmark(load_final_model(), data_yaml, config, device)
    latency.update(measured_at_utc=datetime.now(timezone.utc).isoformat(),
                   cpu_threads=4, seed=42, cudnn_benchmark=False, cudnn_deterministic=True)

    # Primary image labels describe sample coverage; detection GT counts remain unchanged.
    classification = Path(data_root).resolve() / "classification" / "validation"
    classification_paths = sorted(classification.glob("*/*.png"), key=lambda path: path.name)
    primary_counts = {str(index): sum(path.parent.name == name for path in classification_paths)
                      for index, name in enumerate(config["class_order"])}
    assert primary_counts == {str(index): 45 for index in range(6)}
    assert sorted(latency["image_ids"]) == [path.name for path in classification_paths]
    assert len(latency["sample_ms"]) == 270
    result = {
        "protocol": {
            "measured_at_utc": latency["measured_at_utc"],
            "scope": "Preloaded BGR image to CPU prediction: preprocessing, host-device transfers, model, and task postprocessing (NMS/boxes). Excludes disk I/O and model loading.",
            "sampling": "All 270 fixed validation image IDs; 45 per primary class. Detection uses the supplied RGB images; classification folder labels are used only for primary-class coverage metadata.",
            "batch_size": 1, "precision": "FP32", "warmup_iterations": 20,
            "synchronization": "CUDA synchronized before and after each sample" if device != "cpu" else "Synchronous CPU execution",
            "cpu_threads": 4, "seed": 42, "cudnn_benchmark": False,
            "cudnn_deterministic": True, "concurrent_training": False,
            "device": latency["environment"]["device"], "torch": torch.__version__,
            "python": platform.python_version(),
            "limitation": "Task input sizes and output requirements differ; compare end-to-end application cost, not pure architecture speed or factory throughput.",
        },
        "models": [{"experiment": config["detector"],
                    "mean_ms": latency["end_to_end_mean_ms"],
                    "p50_ms": latency["end_to_end_median_ms"],
                    "p95_ms": latency["end_to_end_p95_ms"], **latency,
                    "device": latency["environment"]["device"],
                    "input_size": config["input_size"], "class_counts": primary_counts}],
    }
    summary_path = ROOT / "experiments" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["latency"] = latency
    summary["latency_note"] = "All validation images; one benchmark refreshes summary, latency and deployment evidence; other training must be stopped."
    save_json(summary_path, summary)
    save_json(ROOT / "experiments" / "latency.json", latency)
    save_json(ROOT / "experiments" / "deployment_benchmark.json", result)
    return result


def evaluate(model, data_yaml, config, device):
    output = ROOT / "experiments"
    output.mkdir(exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    official = model.val(
        data=str(data_yaml), split="val", imgsz=config["input_size"], batch=16,
        device=device, half=False, workers=0, plots=True, save_json=False,
        conf=config["evaluation"]["ap_min_confidence"], iou=config["nms_iou_threshold"],
        max_det=config["evaluation"]["max_detections"],
        project=str(output), name="official_validation", exist_ok=True,
    )
    per_class_ap = []
    for row, class_id in enumerate(official.box.ap_class_index):
        per_class_ap.append({
            "class_id": int(class_id), "class_name": config["class_order"][int(class_id)],
            "ap50": float(official.box.ap50[row]), "ap50_95": float(official.box.ap[row]),
            "precision_at_official_max_f1": float(official.box.p[row]),
            "recall_at_official_max_f1": float(official.box.r[row]),
        })
    records = predict_validation(model, data_yaml, config, device)
    save_json(output / "validation_predictions.json", records)
    fixed, details = evaluate_threshold(records, config["confidence_threshold"], config["class_order"],
                                        config["evaluation"]["matching_iou_threshold"])
    sweep = [evaluate_threshold(records, threshold, config["class_order"],
                                config["evaluation"]["matching_iou_threshold"])[0]
             for threshold in config["evaluation"]["thresholds"]]
    failures = select_failure_cases(details, 5)
    manifest = save_visualizations(records, failures, data_yaml, config, output)
    save_json(output / "failure_cases.json", failures)
    save_json(output / "threshold_sweep.json", sweep)
    threshold_plot(sweep, output / "figures" / "confidence_threshold_sweep.png")
    per_class_ap_plot(per_class_ap, output / "figures" / "per_class_ap50.png")
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    summary = {
        "model": config["detector"], "checkpoint": config["checkpoint_file"],
        "selection": selection, "validation_images": len(records),
        "validation_instances": sum(len(record["ground_truth"]) for record in records),
        "official_metrics": {
            "map50": float(official.box.map50), "map50_95": float(official.box.map),
            "macro_precision_at_max_f1": float(official.box.mp),
            "macro_recall_at_max_f1": float(official.box.mr),
            "precision_recall_note": "Ultralytics summary P/R are measured at its maximum mean-F1 confidence operating point, not the configured fixed confidence threshold.",
            "minimum_confidence_for_ap": config["evaluation"]["ap_min_confidence"],
            "per_class": per_class_ap,
        },
        "fixed_threshold_metrics": fixed,
        "matching_protocol": "Predictions sorted by decreasing confidence; match to the highest-IoU unmatched GT of the same class if IoU >= 0.5; each GT matched at most once.",
        "threshold_sweep": sweep, "visualizations": manifest,
        "failure_case_count": len(failures), "failure_cases": failures,
        "parameter_count": parameter_count,
        "checkpoint_size_bytes": (ROOT / config["checkpoint_file"]).stat().st_size,
        "environment": environment(device),
        "latency": None,
        "latency_note": "Pending separate benchmark with other training workloads stopped.",
    }
    export_path = output / "checkpoint_export.json"
    if export_path.exists():
        summary["checkpoint_export"] = json.loads(export_path.read_text(encoding="utf-8"))
    save_json(output / "summary.json", summary)
    print(json.dumps({"map50": summary["official_metrics"]["map50"],
                      "fixed_threshold": fixed["micro"], "visualizations": len(manifest),
                      "failure_cases": len(failures)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to STUDENT_PUBLIC")
    parser.add_argument("--device", default="0")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Measure latency after other training jobs have stopped")
    args = parser.parse_args()
    if args.benchmark_only:
        print(json.dumps(run_benchmark(args.data_root, args.device), indent=2))
    else:
        data_yaml = prepare_dataset(args.data_root)
        config = read_config()
        model = load_final_model()
        evaluate(model, data_yaml, config, args.device)


if __name__ == "__main__":
    main()
