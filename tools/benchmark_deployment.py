"""Measure one Stage at a time with a shared, complete memory-image timing protocol."""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--data-root', type=Path, required=True, help='STUDENT_PUBLIC directory')
    args = parser.parse_args()
    stage = ROOT / 'submission' / f'Team07_Stage{args.stage}'
    sys.path.insert(0, str(stage / 'src'))
    if args.stage == 3:
        from evaluate import run_benchmark
        result = run_benchmark(args.data_root, '0')
        print(json.dumps([{'experiment': row['experiment'], 'mean_ms': row['mean_ms'],
                          'p95_ms': row['p95_ms'], 'samples': row['samples']}
                         for row in result['models']], indent=2))
        return
    config = read(stage / 'config.json')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    hardware = torch.cuda.get_device_name(device)
    # Keep each task's official representation and check paired image identities.
    classification = args.data_root / 'classification'
    classification_paths = sorted((classification / 'validation').glob('*/*.png'), key=lambda p: p.name)
    detection_dir = args.data_root / 'detection' / 'validation' / 'images'
    for path in classification_paths:
        with Image.open(path) as a, Image.open(detection_dir / path.name) as b:
            assert a.size == b.size == (200, 200), path.name

    models = []
    if args.stage == 1:
        from common import DefectDataset, measure_inference
        from model import load_checkpoint
        model, checkpoint = load_checkpoint(stage / config['checkpoint_file'], device)
        raw = measure_inference(model, DefectDataset(classification, 'validation', config), device)
        models.append({'experiment': checkpoint['experiment'], 'mean_ms': raw['mean_ms_per_image'],
                       'p50_ms': raw['median_ms_per_image'], 'p95_ms': raw['p95_ms_per_image'], **raw})
        evaluation = read(stage / 'evaluation/metrics.json')
        evaluation['inference'] = raw
        write(stage / 'evaluation/metrics.json', evaluation)
    elif args.stage == 2:
        from evaluate import benchmark
        from model import load_checkpoint
        from utils import make_loader
        summary = read(stage / 'experiments/summary.json')
        for item in summary['experiments']:
            model, checkpoint = load_checkpoint(stage / item['checkpoint'], device)
            loader = make_loader(classification, 'validation', checkpoint['input_size'], 32)
            raw = benchmark(model, loader, device)
            models.append({'experiment': item['experiment'], **raw})
            if item['experiment'] == summary['final_experiment']:
                evaluation = read(stage / 'evaluation/metrics.json')
                evaluation['latency'] = raw
                write(stage / 'evaluation/metrics.json', evaluation)
            del model
    expected_ids = [path.name for path in classification_paths]
    primary_counts = {str(i): sum(path.parent.name == name for path in classification_paths)
                      for i, name in enumerate(config['class_order'])}
    assert primary_counts == {str(i): 45 for i in range(6)}
    for row in models:
        assert sorted(row['image_ids']) == expected_ids and len(row['sample_ms']) == 270
        if args.stage != 3:
            assert row['class_counts'] == primary_counts
        row.update(device=hardware, precision='FP32', batch_size=1, samples=270,
                   warmup_iterations=20, input_size=config['input_size'], class_counts=primary_counts)
    result = {
        'protocol': {
            'measured_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Preloaded BGR image to CPU prediction: preprocessing, host-device transfers, model, and task postprocessing (class score/argmax or NMS/boxes). Excludes disk I/O and model loading.',
            'sampling': 'All 270 fixed validation image IDs; 45 per class. Classification uses its supplied grayscale images, detection its supplied RGB images; these representations are not pixel-identical.',
            'batch_size': 1, 'precision': 'FP32', 'warmup_iterations': 20,
            'synchronization': 'CUDA synchronized before and after each sample',
            'cpu_threads': 4, 'concurrent_training': False,
            'device': hardware, 'torch': torch.__version__, 'python': platform.python_version(),
            'limitation': 'Task input sizes and output requirements differ; compare end-to-end application cost, not pure architecture speed or factory throughput.',
        }, 'models': models,
    }
    write(stage / 'experiments/deployment_benchmark.json', result)
    print(json.dumps([{'experiment': row['experiment'], 'mean_ms': row['mean_ms'],
                      'p95_ms': row['p95_ms'], 'samples': row['samples']} for row in models], indent=2))


if __name__ == '__main__':
    main()
