"""Evaluate the local final checkpoint; produces validation metrics and evidence."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import CLASS_ORDER, load_checkpoint
from utils import make_loader, plot_confusion, seed_everything, validate, write_json

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def benchmark(model, loader, device):
    model.eval()
    images = []
    for path, _ in loader.dataset.samples:
        with Image.open(path) as source:
            images.append(np.asarray(source.convert('RGB'))[:, :, ::-1].copy())

    def predict(image):
        converted = Image.fromarray(image[:, :, ::-1])
        tensor = loader.dataset.transform(converted).unsqueeze(0).to(device)
        scores, indices = model(tensor).softmax(1).max(1)
        return indices.cpu().tolist(), scores.cpu().tolist()

    for index in range(20):
        predict(images[index % len(images)])
    elapsed = []
    for image in images:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict(image)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
    return {'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'batch_size': 1, 'precision': 'FP32', 'warmup': 20, 'iterations': len(images),
        'mean_ms': float(np.mean(elapsed)), 'p50_ms': float(np.median(elapsed)), 'p95_ms': float(np.percentile(elapsed, 95)),
        'image_ids': [Path(path).name for path, _ in loader.dataset.samples],
        'class_counts': {str(label): sum(y == label for _, y in loader.dataset.samples) for label in range(6)},
        'sample_ms': elapsed,
        'scope': 'Memory BGR image to CPU class/score: color conversion, resize/normalization, host-device transfer, forward and softmax/argmax; excludes disk I/O and model loading.'}


def failure_visualization(samples, labels, predictions, probabilities, output):
    failures = [(i, max(probabilities[i])) for i, (a, b) in enumerate(zip(labels, predictions)) if a != b]
    failures.sort(key=lambda item: item[1], reverse=True)
    chosen = failures[:6]
    if not chosen:
        return []
    canvas = Image.new('RGB', (600, 260 * ((len(chosen) + 2) // 3)), 'white')
    draw = ImageDraw.Draw(canvas)
    records = []
    for cell, (idx, confidence) in enumerate(chosen):
        x, y = (cell % 3) * 200, (cell // 3) * 260
        with Image.open(samples[idx][0]) as image:
            canvas.paste(image.convert('RGB').resize((200, 200)), (x, y))
        draw.text((x + 4, y + 203), f'True: {CLASS_ORDER[labels[idx]]}', fill='black')
        draw.text((x + 4, y + 220), f'Pred: {CLASS_ORDER[predictions[idx]]}', fill='black')
        draw.text((x + 4, y + 237), f'Confidence: {confidence:.3f}', fill='black')
        records.append({'file': Path(samples[idx][0]).name, 'true': CLASS_ORDER[labels[idx]],
                        'predicted': CLASS_ORDER[predictions[idx]], 'confidence': confidence})
    canvas.save(output)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'stage2_model.pth')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    seed_everything(42)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    loader = make_loader(args.data, 'validation', checkpoint['input_size'], 32)
    metrics, labels, predictions, probabilities = validate(model, loader, device)
    plot_confusion(metrics['confusion_matrix'], args.output / 'confusion_matrix.png', 'Final Stage 2 model')
    failures = failure_visualization(loader.dataset.samples, labels, predictions, probabilities, args.output / 'failure_cases.png')
    result = {'architecture': checkpoint['architecture'], 'strategy': checkpoint['strategy'],
              'metrics': metrics, 'latency': benchmark(model, loader, device), 'failure_cases': failures}
    write_json(args.output / 'metrics.json', result)
    write_json(args.output / 'predictions.json', [
        {'file': str(Path(p).relative_to(args.data)), 'true': int(y), 'predicted': int(pred), 'probabilities': probs}
        for (p, _), y, pred, probs in zip(loader.dataset.samples, labels, predictions, probabilities)])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
