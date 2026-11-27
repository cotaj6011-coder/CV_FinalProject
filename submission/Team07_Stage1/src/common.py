"""Dataset, measurement, and evidence output shared by training and evaluation."""

import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset
from torchvision import transforms


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed):
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DefectDataset(Dataset):
    def __init__(self, root, split, config, augment=False):
        self.samples = [
            (path, index)
            for index, name in enumerate(config["class_order"])
            for path in sorted((Path(root) / split / name).glob("*.png"))
        ]
        operations = [transforms.Resize(tuple(config["input_size"]))]
        if augment:
            settings = config["augmentation"]
            operations += [
                transforms.RandomHorizontalFlip(settings["horizontal_flip_p"]),
                transforms.RandomVerticalFlip(settings["vertical_flip_p"]),
                transforms.RandomRotation(settings["rotation_degrees"]),
            ]
        operations += [
            transforms.ToTensor(),
            transforms.Normalize(config["normalization_mean"], config["normalization_std"]),
        ]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = self.transform(source.convert("L"))
        return image, label, str(path)


def classification_metrics(labels, predictions, class_order):
    report = classification_report(
        labels, predictions, labels=list(range(len(class_order))),
        target_names=class_order, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "samples": len(labels),
        "per_class": {name: report[name] for name in class_order},
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(class_order)))
        ).tolist(),
    }


def evaluate_model(model, loader, device, class_order, max_batches=None):
    model.eval()
    labels, predictions, paths, confidences = [], [], [], []
    loss_total = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for batch, (images, targets, batch_paths) in enumerate(loader):
            if max_batches is not None and batch >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss_total += float(criterion(logits, targets))
            probabilities = logits.softmax(dim=1)
            scores, indices = probabilities.max(dim=1)
            labels.extend(targets.cpu().tolist())
            predictions.extend(indices.cpu().tolist())
            confidences.extend(scores.cpu().tolist())
            paths.extend(batch_paths)
    result = classification_metrics(labels, predictions, class_order)
    result["loss"] = loss_total / len(labels)
    records = [
        {"image": str(Path(path).name), "class_folder": class_order[label],
         "true_label": label, "prediction": prediction, "confidence": confidence}
        for path, label, prediction, confidence in zip(paths, labels, predictions, confidences)
    ]
    return result, records, paths


def save_confusion(matrix, class_order, output, title):
    figure, axes = plt.subplots(figsize=(8, 7))
    axes.imshow(matrix, cmap="Blues")
    axes.set(xticks=range(len(class_order)), yticks=range(len(class_order)),
             xticklabels=class_order, yticklabels=class_order,
             xlabel="Predicted", ylabel="Ground truth", title=title)
    plt.setp(axes.get_xticklabels(), rotation=35, ha="right")
    for row in range(len(class_order)):
        for column in range(len(class_order)):
            value = matrix[row][column]
            axes.text(column, row, str(value), ha="center", va="center",
                      color="white" if value > np.max(matrix) / 2 else "black")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_curves(history, output, title):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train")
        axis.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.title())
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_failures(records, paths, class_order, output, limit=12):
    indices = [index for index, record in enumerate(records)
               if record["true_label"] != record["prediction"]]
    indices.sort(key=lambda index: records[index]["confidence"], reverse=True)
    selected = indices[:limit]
    if not selected:
        return []
    figure, axes = plt.subplots((len(selected) + 3) // 4, 4,
                               figsize=(12, 3.0 * ((len(selected) + 3) // 4)), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, selected):
        record = records[index]
        with Image.open(paths[index]) as source:
            axis.imshow(source.convert("L"), cmap="gray", vmin=0, vmax=255)
        axis.set_title(f'{record["image"]}\nGT: {class_order[record["true_label"]]}\n'
                       f'Pred: {class_order[record["prediction"]]} ({record["confidence"]:.2f})', fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return [records[index] for index in selected]


def measure_inference(model, dataset, device):
    """Time the complete memory-image classification pipeline on every validation image."""
    model.eval()
    images = []
    for path, _ in dataset.samples:
        with Image.open(path) as source:
            images.append(np.asarray(source.convert('RGB'))[:, :, ::-1].copy())

    def predict(image):
        converted = Image.fromarray(image[:, :, ::-1]).convert('L')
        tensor = dataset.transform(converted).unsqueeze(0).to(device)
        scores, indices = model(tensor).softmax(1).max(1)
        return indices.cpu().tolist(), scores.cpu().tolist()

    elapsed = []
    with torch.inference_mode():
        for index in range(20):
            predict(images[index % len(images)])
        for image in images:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            predict(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - started) * 1000)
    return {
        "mean_ms_per_image": float(np.mean(elapsed)),
        "median_ms_per_image": float(np.median(elapsed)),
        "p95_ms_per_image": float(np.percentile(elapsed, 95)),
        "batch_size": 1, "precision": "FP32", "warmup_iterations": 20, "measured_images": len(elapsed),
        "includes": "Memory BGR image to CPU class/score: color conversion, resize/normalization, host-device transfer, forward and softmax/argmax; excludes disk I/O and model loading",
        "image_ids": [Path(path).name for path, _ in dataset.samples],
        "class_counts": {str(label): sum(y == label for _, y in dataset.samples) for label in range(6)},
        "sample_ms": elapsed,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
