"""Dataset, reproducibility, validation metrics and plotting utilities."""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CLASS_ORDER, MEAN, STD


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def make_loader(data_root, split, input_size, batch_size, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(Path(data_root) / split, transform=transform)
    assert dataset.classes == CLASS_ORDER, dataset.classes
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def classification_metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=range(6), average='macro', zero_division=0)
    return {'accuracy': accuracy_score(labels, predictions), 'macro_precision': precision,
            'macro_recall': recall, 'macro_f1': f1,
            'confusion_matrix': confusion_matrix(labels, predictions, labels=range(6)).tolist(),
            'per_class': classification_report(labels, predictions, labels=range(6),
                target_names=CLASS_ORDER, output_dict=True, zero_division=0)}


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    labels, predictions, probabilities = [], [], []
    total_loss = 0.0
    for images, targets in loader:
        logits = model(images.to(device))
        total_loss += torch.nn.functional.cross_entropy(logits, targets.to(device), reduction='sum').item()
        labels.extend(targets.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        probabilities.extend(logits.softmax(1).cpu().tolist())
    metrics = classification_metrics(labels, predictions)
    metrics['loss'] = total_loss / len(labels)
    return metrics, labels, predictions, probabilities


def plot_confusion(matrix, output, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
           xlabel='Predicted', ylabel='True', title=title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                    color='white' if matrix[i][j] > np.max(matrix) / 2 else 'black')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_history(history, output, title):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    x = [h['epoch'] for h in history]
    for ax, metric, ylabel in zip(axes[:2], ['loss', 'accuracy'], ['Cross entropy', 'Accuracy']):
        ax.plot(x, [h['train_' + metric] for h in history], label='Train')
        ax.plot(x, [h['val_' + metric] for h in history], label='Validation')
        ax.set(xlabel='Epoch', ylabel=ylabel)
        ax.legend()
    axes[2].plot(x, [h['val_macro_f1'] for h in history])
    axes[2].set(xlabel='Epoch', ylabel='Validation macro F1')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
