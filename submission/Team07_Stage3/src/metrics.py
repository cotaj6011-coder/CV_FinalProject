"""Confidence-ordered, class-aware one-to-one matching at a fixed IoU."""
import numpy as np


def box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_first + area_second - intersection)


def match_image(record, confidence, iou_threshold=0.5):
    ground_truth = record["ground_truth"]
    predictions = sorted(
        (p for p in record["predictions"] if p["confidence"] >= confidence),
        key=lambda p: p["confidence"], reverse=True)
    unmatched = set(range(len(ground_truth)))
    matches, false_positives = [], []
    for prediction in predictions:
        candidates = [(box_iou(prediction["bbox"], gt["bbox"]), index)
                      for index, gt in enumerate(ground_truth)
                      if index in unmatched and gt["class_id"] == prediction["class_id"]]
        overlap, index = max(candidates, default=(0.0, -1))
        if overlap >= iou_threshold:
            matches.append({"prediction": prediction, "ground_truth_index": index, "iou": overlap})
            unmatched.remove(index)
        else:
            same_class = [box_iou(prediction["bbox"], gt["bbox"]) for gt in ground_truth
                          if gt["class_id"] == prediction["class_id"]]
            maximum = max(same_class, default=0.0)
            false_positives.append({
                **prediction, "best_same_class_iou": maximum,
                "kind": "poor_localization" if 0.1 <= maximum < iou_threshold else "false_positive",
            })
    return {
        "image": record["image"], "matches": matches,
        "false_positives": false_positives,
        "false_negatives": [ground_truth[index] for index in sorted(unmatched)],
    }


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_threshold(records, confidence, class_order, iou_threshold=0.5):
    counts = np.zeros((len(class_order), 3), dtype=int)
    details = []
    for record in records:
        detail = match_image(record, confidence, iou_threshold)
        details.append(detail)
        for match in detail["matches"]:
            counts[match["prediction"]["class_id"], 0] += 1
        for prediction in detail["false_positives"]:
            counts[prediction["class_id"], 1] += 1
        for truth in detail["false_negatives"]:
            counts[truth["class_id"], 2] += 1
    per_class = [{"class_id": index, "class_name": name,
                  **precision_recall(*map(int, counts[index]))}
                 for index, name in enumerate(class_order)]
    summary = {
        "confidence_threshold": confidence, "matching_iou_threshold": iou_threshold,
        "micro": precision_recall(*map(int, counts.sum(axis=0))),
        "macro": {key: float(np.mean([row[key] for row in per_class]))
                  for key in ("precision", "recall", "f1")},
        "per_class": per_class,
    }
    return summary, details


def select_failure_cases(details, count=5):
    pools = [
        [d for d in details if any(p["kind"] == "poor_localization" for p in d["false_positives"])],
        [d for d in details if d["false_negatives"]],
        [d for d in details if any(p["kind"] == "false_positive" for p in d["false_positives"])],
    ]
    selected, used, covered_classes = [], set(), set()
    while len(selected) < count:
        advanced = False
        for pool in pools:
            available = [item for item in pool if item["image"] not in used]
            if available and len(selected) < count:
                def error_classes(item):
                    return {error["class_id"] for error in item["false_positives"] + item["false_negatives"]}
                # Preserve filename order while broadening error-class coverage.
                item = next((item for item in available if error_classes(item) - covered_classes), available[0])
                selected.append(item)
                used.add(item["image"])
                covered_classes.update(error_classes(item))
                advanced = True
        if not advanced:
            break
    return selected
