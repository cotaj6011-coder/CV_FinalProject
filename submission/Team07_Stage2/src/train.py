"""Run the fixed 2-family x 2-strategy comparison on the supplied split."""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import torch

from model import CLASS_ORDER, MEAN, STD, SOURCES, build_model, classifier, configure_strategy, load_checkpoint
from utils import make_loader, plot_confusion, plot_history, seed_everything, validate, write_json

ROOT = Path(__file__).resolve().parents[1]
SELECTION_RULE = ('Highest validation macro F1; then smaller checkpoint; '
                  'then fewer trainable parameters; then experiment name ascending.')


def select_best_experiment(results):
    return min(results, key=lambda result: (-result['metrics']['macro_f1'],
        result['checkpoint_bytes'], result['trainable_parameters'], result['experiment']))


def train_experiment(args, architecture, strategy):
    seed_everything(args.seed)
    device = torch.device(args.device)
    name = architecture + '__' + strategy
    output = ROOT / 'experiments' / name
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(architecture, pretrained=True).to(device)
    configure_strategy(model, architecture, strategy)
    head_ids = {id(p) for p in classifier(model, architecture).parameters()}
    groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) in head_ids], 'lr': 1e-3}]
    if strategy == 'full_fine_tuning':
        groups.append({'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': 1e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = make_loader(args.data, 'train', args.input_size, args.batch_size, shuffle=True)
    val_loader = make_loader(args.data, 'validation', args.input_size, args.batch_size)
    history, best_f1, best_epoch = [], -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if strategy == 'feature_extraction':
            model.eval()  # Freeze backbone BatchNorm running statistics as well as weights.
            classifier(model, architecture).train()
        else:
            model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            count += len(targets)
        metrics, _, _, _ = validate(model, val_loader, device)
        row = {'epoch': epoch, 'train_loss': loss_sum / count, 'train_accuracy': correct / count,
               'val_loss': metrics['loss'], 'val_accuracy': metrics['accuracy'], 'val_macro_f1': metrics['macro_f1'],
               'head_lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        print(f"{name} epoch {epoch}/{args.epochs}: train={correct/count:.4f}, val F1={metrics['macro_f1']:.4f}", flush=True)
        if metrics['macro_f1'] > best_f1:
            best_f1, best_epoch = metrics['macro_f1'], epoch
            torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'architecture': architecture, 'strategy': strategy, 'class_order': CLASS_ORDER,
                'input_size': args.input_size, 'normalization_mean': MEAN, 'normalization_std': STD,
                'seed': args.seed, 'best_epoch': best_epoch}, output / 'best.pth')
        scheduler.step()
    training_seconds = time.perf_counter() - start
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    del model, optimizer
    model, _ = load_checkpoint(output / 'best.pth', device)
    metrics, labels, predictions, probabilities = validate(model, val_loader, device)
    result = {'experiment': name, 'architecture': architecture, 'strategy': strategy,
        'pretrained_source': SOURCES[architecture], 'epochs': args.epochs, 'best_epoch': best_epoch,
        'seed': args.seed, 'input_size': args.input_size, 'batch_size': args.batch_size,
        'optimizer': 'AdamW', 'head_learning_rate': 1e-3,
        'backbone_learning_rate': 0 if strategy == 'feature_extraction' else 1e-4,
        'weight_decay': 1e-4, 'scheduler': 'CosineAnnealingLR', 'augmentation': 'none',
        'parameters': parameters, 'trainable_parameters': trainable, 'trainable_layers': trainable_names,
        'checkpoint_bytes': (output / 'best.pth').stat().st_size, 'training_seconds': training_seconds,
        'metrics': metrics, 'checkpoint': f'experiments/{name}/best.pth'}
    write_json(output / 'metrics.json', result)
    write_json(output / 'history.json', history)
    write_json(output / 'predictions.json', [
        {'file': str(Path(p).relative_to(Path(args.data))), 'true': int(y), 'predicted': int(pred), 'probabilities': probs}
        for (p, _), y, pred, probs in zip(val_loader.dataset.samples, labels, predictions, probabilities)])
    with (output / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output / 'curves.png', name)
    plot_confusion(metrics['confusion_matrix'], output / 'confusion_matrix.png', name)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True, help='Path to classification/ containing train/ and validation/')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--architecture', choices=['all', 'resnet18', 'mobilenet_v3_small'], default='all')
    parser.add_argument('--strategy', choices=['all', 'feature_extraction', 'full_fine_tuning'], default='all')
    args = parser.parse_args()
    team_id = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))['team_id']
    architectures = ['resnet18', 'mobilenet_v3_small'] if args.architecture == 'all' else [args.architecture]
    strategies = ['feature_extraction', 'full_fine_tuning'] if args.strategy == 'all' else [args.strategy]
    for architecture in architectures:
        for strategy in strategies:
            train_experiment(args, architecture, strategy)
    results = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT / 'experiments').glob('*/metrics.json'))]
    best = select_best_experiment(results)
    shutil.copyfile(ROOT / best['checkpoint'], ROOT / 'stage2_model.pth')
    config = {'team_id': team_id, 'stage': 2, 'architecture': best['architecture'], 'strategy': best['strategy'],
        'pretrained_source': best['pretrained_source'], 'input_size': best['input_size'], 'class_order': CLASS_ORDER,
        'normalization_mean': MEAN, 'normalization_std': STD, 'seed': best['seed'], 'checkpoint_file': 'stage2_model.pth',
        'selection_metric': 'validation_macro_f1', 'selection_rule': SELECTION_RULE,
        'experiment': best['experiment'], 'best_epoch': best['best_epoch']}
    write_json(ROOT / 'config.json', config)
    write_json(ROOT / 'experiments' / 'summary.json', {'experiments': results, 'final_experiment': best['experiment'],
        'selection_rule': SELECTION_RULE})


if __name__ == '__main__':
    main()
