"""Save report figures from real validation predictions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def detection_pair(image_path, record, confidence, class_order, destination):
    image = Image.open(image_path).convert("RGB")
    predictions = [p for p in record["predictions"] if p["confidence"] >= confidence]
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, objects, title, color in zip(
            axes, (record["ground_truth"], predictions),
            ("Ground truth", f"Prediction (confidence >= {confidence:.2f})"), ("lime", "orange")):
        axis.imshow(image)
        label_positions = []
        for item in objects:
            x1, y1, x2, y2 = item["bbox"]
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, edgecolor=color, linewidth=1.6))
            label = class_order[item["class_id"]]
            if "confidence" in item:
                label += f" {item['confidence']:.2f}"
            label_x = min(x1 + 1, image.width - 68)
            label_y = min(y1 + 1, image.height - 9)
            while any(abs(label_y - previous_y) < 8 and abs(label_x - previous_x) < 65
                      for previous_x, previous_y in label_positions):
                label_y += 8
            label_positions.append((label_x, label_y))
            axis.text(label_x, label_y, label, fontsize=6.5, color="white",
                      bbox={"facecolor": "black", "alpha": 0.7, "pad": 1},
                      verticalalignment="top", clip_on=True)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle(record["image"], fontsize=13)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def threshold_plot(rows, destination):
    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        axis.plot([row["confidence_threshold"] for row in rows],
                  [row["micro"][key] for row in rows], marker="o", label=label)
    axis.set(xlabel="Confidence threshold", ylabel="Micro metric (IoU >= 0.5)", ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def per_class_ap_plot(rows, destination):
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar([row["class_name"] for row in rows], [row["ap50"] for row in rows])
    axis.set(ylabel="AP@0.5", ylim=(0, 1.0))
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
