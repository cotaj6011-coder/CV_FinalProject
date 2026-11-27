"""Run the traditional baseline and the four controlled CNN experiments."""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from common import (DefectDataset, classification_metrics, evaluate_model,
                    measure_inference, save_confusion, save_curves, save_failures,
                    set_seed, write_json)
from model import SurfaceCNN, load_checkpoint


STAGE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "E1": {"augmentation": False, "dropout": 0.0, "methods": []},
    "E2": {"augmentation": True, "dropout": 0.0, "methods": ["data_augmentation"]},
    "E3": {"augmentation": False, "dropout": 0.5, "methods": ["dropout"]},
    "E4": {"augmentation": True, "dropout": 0.5, "methods": ["data_augmentation", "dropout"]},
}


def train_epoch(model, loader, optimizer, device, max_batches=None):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for batch, (images, labels, _) in enumerate(loader):
        if max_batches is not None and batch >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum())
        loss_sum += float(loss.detach()) * len(labels)
    return loss_sum / total, correct / total


def run_cnn(name, config, data_root, output_root, device, smoke=False):
    settings = EXPERIMENTS[name]
    set_seed(config["seed"])
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)
    training = DefectDataset(data_root, "train", config, augment=settings["augmentation"])
    validation = DefectDataset(data_root, "validation", config)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True,
                              generator=generator, num_workers=config["num_workers"])
    val_loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"])
    model = SurfaceCNN(len(config["class_order"]), settings["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    history, best_f1, best_epoch = [], -1.0, 0
    started = time.perf_counter()
    epoch_count = 1 if smoke else config["epochs"]
    for epoch in range(1, epoch_count + 1):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, device, max_batches=2 if smoke else None)
        validation_metrics, _, _ = evaluate_model(
            model, val_loader, device, config["class_order"], max_batches=2 if smoke else None)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_accuracy": train_accuracy, "val_loss": validation_metrics["loss"],
                        "val_accuracy": validation_metrics["accuracy"],
                        "val_macro_f1": validation_metrics["macro_f1"]})
        if validation_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = validation_metrics["macro_f1"], epoch
            torch.save({
                "state_dict": {key: tensor.cpu() for key, tensor in model.state_dict().items()},
                "model_name": "SurfaceCNN", "class_order": config["class_order"],
                "input_size": config["input_size"], "normalization_mean": config["normalization_mean"],
                "normalization_std": config["normalization_std"], "dropout": settings["dropout"],
                "improvement_methods": settings["methods"], "experiment": name,
                "seed": config["seed"], "best_epoch": best_epoch, "epochs_budget": epoch_count,
            }, output / "best_model.pth")
        write_json(output / "history.json", history)
        print(f'{name} {epoch:02d}/{epoch_count} train_loss={train_loss:.4f} '
              f'val_loss={validation_metrics["loss"]:.4f} '
              f'val_acc={validation_metrics["accuracy"]:.4f} '
              f'val_f1={validation_metrics["macro_f1"]:.4f}', flush=True)
    duration = time.perf_counter() - started
    model, _ = load_checkpoint(output / "best_model.pth", device)
    metrics, predictions, paths = evaluate_model(model, val_loader, device, config["class_order"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics.update({
        "experiment": name, "model_name": "SurfaceCNN", "seed": config["seed"],
        "best_epoch": best_epoch, "epochs_budget": epoch_count,
        "train_samples": len(training), "validation_samples": len(validation),
        "improvement_methods": settings["methods"], "dropout": settings["dropout"],
        "parameter_count": parameter_count, "trainable_parameter_count": parameter_count,
        "checkpoint_bytes": (output / "best_model.pth").stat().st_size,
        "training_seconds": duration, "smoke_only": smoke,
        "inference": measure_inference(model, validation, device),
    })
    write_json(output / "metrics.json", metrics)
    write_json(output / "predictions.json", predictions)
    save_confusion(metrics["confusion_matrix"], config["class_order"], output / "confusion_matrix.png", name)
    save_curves(history, output / "training_curves.png", f'{name}: {" + ".join(settings["methods"]) or "Baseline"}')
    failures = save_failures(predictions, paths, config["class_order"], output / "failure_cases.png")
    write_json(output / "failure_cases.json", failures)
    return metrics


def traditional_features(data_root, split, config):
    features, labels, paths = [], [], []
    size = tuple(config["traditional"]["input_size"])
    for label, class_name in enumerate(config["class_order"]):
        for path in sorted((data_root / split / class_name).glob("*.png")):
            with Image.open(path) as source:
                image = source.convert("L").resize(size, resample=Image.Resampling.BILINEAR)
                features.append(np.asarray(image, dtype=np.float32).ravel() / 255.0)
            labels.append(label)
            paths.append(path)
    return np.asarray(features), np.asarray(labels), paths


def run_traditional(config, data_root, output_root):
    set_seed(config["seed"])
    output = output_root / "traditional"
    output.mkdir(parents=True, exist_ok=True)
    train_x, train_y, _ = traditional_features(data_root, "train", config)
    val_x, val_y, paths = traditional_features(data_root, "validation", config)
    settings = config["traditional"]
    model = Pipeline([
        ("pca", PCA(n_components=settings["pca_components"], whiten=settings["whiten"],
                    svd_solver="randomized", random_state=config["seed"])),
        ("svm", SVC(C=settings["svm_C"], kernel=settings["svm_kernel"], gamma=settings["svm_gamma"])),
    ])
    started = time.perf_counter()
    model.fit(train_x, train_y)
    duration = time.perf_counter() - started
    predictions = model.predict(val_x)
    decision_values = model.decision_function(val_x)
    scores = np.max(decision_values, axis=1)
    metrics = classification_metrics(val_y.tolist(), predictions.tolist(), config["class_order"])
    joblib.dump(model, output / "pca_svm.joblib")
    for _ in range(10):
        model.predict(val_x[:1])
    latencies = []
    for index in range(100):
        started = time.perf_counter()
        model.predict(val_x[index:index + 1])
        latencies.append((time.perf_counter() - started) * 1000)
    metrics.update({"experiment": "traditional", "model_name": "PCA + RBF SVM", "seed": config["seed"],
                    "train_samples": len(train_y), "validation_samples": len(val_y),
                    "training_seconds": duration, "checkpoint_bytes": (output / "pca_svm.joblib").stat().st_size,
                    "pca_explained_variance_ratio": float(model["pca"].explained_variance_ratio_.sum()),
                    "support_vectors": int(model["svm"].n_support_.sum()),
                    "settings": settings,
                    "inference": {"mean_ms_per_image": float(np.mean(latencies)),
                                  "median_ms_per_image": float(np.median(latencies)),
                                  "p95_ms_per_image": float(np.percentile(latencies, 95)),
                                  "batch_size": 1, "warmup_iterations": 10, "measured_images": 100,
                                  "device": "cpu", "includes": "PCA transform and SVM prediction; excludes image preprocessing"}})
    records = [{"image": path.name, "class_folder": config["class_order"][int(label)],
                "true_label": int(label), "prediction": int(prediction), "decision_score": float(score)}
               for path, label, prediction, score in zip(paths, val_y, predictions, scores)]
    write_json(output / "metrics.json", metrics)
    write_json(output / "predictions.json", records)
    save_confusion(metrics["confusion_matrix"], config["class_order"], output / "confusion_matrix.png", "PCA + RBF SVM")
    print(f'Traditional baseline Macro F1={metrics["macro_f1"]:.4f}', flush=True)
    return metrics


def finalize(config, output_root):
    results = {name: json.loads((output_root / name / "metrics.json").read_text(encoding="utf-8"))
               for name in ["traditional", *EXPERIMENTS]}
    chosen = max(config["selection_candidates"], key=lambda name: results[name]["macro_f1"])
    selected = results[chosen]
    shutil.copyfile(output_root / chosen / "best_model.pth", STAGE_ROOT / config["checkpoint_file"])
    config.update({"selected_experiment": chosen, "dropout": selected["dropout"],
                   "improvement_methods": selected["improvement_methods"],
                   "best_epoch": selected["best_epoch"]})
    write_json(STAGE_ROOT / "config.json", config)
    summary = {
        "stage": 1, "seed": config["seed"], "class_order": config["class_order"],
        "selection_metric": "validation_macro_f1", "selection_candidates": config["selection_candidates"],
        "selected_experiment": chosen, "selected_model": selected, "experiments": results,
        "data_note": "The supplied fixed split contains an identical patches image pair: train TRN_0504.png and validation VAL_0101.png. The required split was preserved; validation metrics may be slightly optimistic.",
    }
    write_json(output_root / "summary.json", summary)
    columns = ["experiment", "accuracy", "macro_precision", "macro_recall", "macro_f1", "training_seconds", "checkpoint_bytes"]
    with (output_root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results.values())
    baseline = results["E1"]["macro_f1"]
    notes = ["# Stage 1 實驗紀錄", "", "## 實驗設計", "",
             "固定課程 train/validation split，seed=42。CNN 僅用灰階影像 resize 128×128、tensor、固定 mean/std=0.5。",
             "自製三層 Conv(32/64/128) + ReLU + MaxPool，AdaptiveAvgPool(4×4)、Linear128、ReLU、Dropout、Linear6；所有權重從零訓練。",
             "A 為資料擴增（水平/垂直翻轉各 p=0.5、旋轉 ±10°）；B 為 classifier Dropout p=0.5。所有 CNN 使用 Adam lr=0.001、weight_decay=0、batch_size=64、40 epochs；不使用 early stopping 或 learning-rate scheduler。",
             "每組依 validation Macro F1 選 epoch，完整訓練40 epochs。最終 improved CNN 僅從 E2/E3/E4 選 Macro F1 最高者；tie 保留最早 epoch/較前實驗。",
             "傳統方法以64×64灰階像素/255作特徵，PCA100(whiten=True) + RBF SVM(C=10,gamma=scale)；PCA、SVM均只fit training。", "", "## 結果", "",
             "| 實驗 | Accuracy | Macro Precision | Macro Recall | Macro F1 |", "|---|---:|---:|---:|---:|"]
    for name, result in results.items():
        notes.append(f'| {name} | {result["accuracy"]:.4f} | {result["macro_precision"]:.4f} | {result["macro_recall"]:.4f} | {result["macro_f1"]:.4f} |')
    notes += ["", f'最終選擇 {chosen}，best epoch={selected["best_epoch"]}，Macro F1={selected["macro_f1"]:.4f}。',
              f'A only、B only、A+B 相對baseline的Macro F1差為 {results["E2"]["macro_f1"]-baseline:+.4f}、{results["E3"]["macro_f1"]-baseline:+.4f}、{results["E4"]["macro_f1"]-baseline:+.4f}。',
              "", "## 訓練曲線與泛化", "",
              "| 實驗 | 最佳 epoch | 最佳 epoch train/val accuracy | 最終 epoch train/val accuracy | 最終 epoch train/val loss |",
              "|---|---:|---|---|---|"]
    for name in EXPERIMENTS:
        history = json.loads((output_root / name / "history.json").read_text(encoding="utf-8"))
        best = history[results[name]["best_epoch"] - 1]
        last = history[-1]
        notes.append(f'| {name} | {results[name]["best_epoch"]} | {best["train_accuracy"]:.4f}/{best["val_accuracy"]:.4f} | '
                     f'{last["train_accuracy"]:.4f}/{last["val_accuracy"]:.4f} | {last["train_loss"]:.4f}/{last["val_loss"]:.4f} |')
        experiment_config = dict(config)
        experiment_config.update({"experiment": name, "dropout": EXPERIMENTS[name]["dropout"],
                                  "improvement_methods": EXPERIMENTS[name]["methods"],
                                  "augmentation_enabled": EXPERIMENTS[name]["augmentation"],
                                  "best_epoch": results[name]["best_epoch"],
                                  "checkpoint_file": "best_model.pth"})
        experiment_config.pop("selected_experiment", None)
        write_json(output_root / name / "experiment_config.json", experiment_config)
    matrix = selected["confusion_matrix"]
    confusions = sorted(((matrix[row][column], row, column)
                         for row in range(len(config["class_order"]))
                         for column in range(len(config["class_order"]))
                         if row != column and matrix[row][column]), reverse=True)
    notes += ["", "最終模型的非對角誤判（由多至少）："]
    notes += [f'- {config["class_order"][row]} → {config["class_order"][column]}：{count} 張。'
              for count, row, column in confusions]
    notes += [
              "", "## 解讀界線", "",
              "此為固定驗證集上的模型選擇結果，未宣稱獨立測試成績或統計顯著差異。所有設定共同依規定 seed=42，未進行多seed實驗。",
              "課程固定split中有一對完全相同影像：train/patches/TRN_0504.png 與 validation/patches/VAL_0101.png。遵守固定split保留，驗證成績可能稍樂觀。",
              "CNN訓練accuracy/loss為訓練模式、可能含augmentation/dropout；validation為eval模式，因此兩者差距不能完全等同過擬合。",
              "CNN延遲使用全部Validation影像，batch=1、FP32、20次暖機；包含記憶體影像的色彩轉換、前處理、裝置傳輸、forward、softmax/argmax與結果回CPU，排除磁碟讀圖及模型載入。SVM計時為CPU PCA+predict；不同裝置與計時範圍不能當純架構速度比較。",
              "各組 history.json/training_curves.png 顯示 train/validation loss與accuracy；confusion_matrix.png、failure_cases.json/png 提供真實驗證失敗例，信心分數不等同錯誤成因證明。"]
    (STAGE_ROOT / "report_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f'Final improved model: {chosen}, Macro F1={selected["macro_f1"]:.4f}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="classification folder containing train/validation")
    parser.add_argument("--config", type=Path, default=STAGE_ROOT / "config.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiment", choices=["all", "traditional", *EXPERIMENTS, "finalize"], default="all")
    parser.add_argument("--smoke", action="store_true", help="Two training batches and one epoch; writes only experiments/smoke")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    output = STAGE_ROOT / "experiments"
    if args.smoke:
        output /= "smoke"
    output.mkdir(parents=True, exist_ok=True)
    if args.experiment in ("all", "traditional") and not args.smoke:
        run_traditional(config, args.data_root, output)
    for name in EXPERIMENTS:
        if args.experiment in ("all", name):
            run_cnn(name, config, args.data_root, output, device, smoke=args.smoke)
    if args.experiment in ("all", "finalize") and not args.smoke:
        finalize(config, output)


if __name__ == "__main__":
    main()
