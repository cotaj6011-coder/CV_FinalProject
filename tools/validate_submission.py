"""Read-only FactoryVision delivery checks. No model loading, GPU use or file writes."""
import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image
from pypdf import PdfReader


CLASS_ORDER = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
COMMON_FIELDS = {'team_id', 'stage', 'input_size', 'class_order', 'seed', 'checkpoint_file'}
CONFIG_FIELDS = {
    1: COMMON_FIELDS | {'model_name', 'normalization_mean', 'normalization_std', 'improvement_methods'},
    2: COMMON_FIELDS | {'architecture', 'strategy', 'pretrained_source', 'normalization_mean', 'normalization_std'},
    3: COMMON_FIELDS | {'detector', 'pretrained_source', 'confidence_threshold', 'nms_iou_threshold', 'bbox_format'},
}
EXCLUDED_PARTS = {'.venv', 'venv', '__pycache__', '.git', '.cache', '.pytest_cache', 'prepared_data', 'data', 'smoke'}
PATH_KEYS = {'checkpoint_file', 'checkpoint', 'source_checkpoint', 'training_results_csv', 'figure'}
PERSONAL_PATH = re.compile(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]|/(?:home|Users)/')
CLASSIFICATION_METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')
errors, pending_metadata = [], []
required_files = set()
counts = {'checks': 0, 'json_files': 0, 'image_files': 0, 'pdf_files': 0, 'zip_files': 0}


def check(condition, message):
    counts['checks'] += 1
    if not condition:
        errors.append(message)
    return bool(condition)


def require_file(path):
    required_files.add(path.resolve())
    return check(path.is_file() and path.stat().st_size > 0, f'Missing or empty file: {path}')


def read_json(path):
    if not require_file(path):
        return None
    try:
        def reject_constant(value):
            raise ValueError(f'Non-finite JSON number: {value}')
        result = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
        counts['json_files'] += 1
        return result
    except (ValueError, UnicodeError) as error:
        check(False, f'Invalid JSON {path}: {error}')
        return None


def probability(value):
    return isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1


def classification_metrics(metrics, label):
    if not check(isinstance(metrics, dict), f'{label}: metrics must be an object'):
        return
    for key in CLASSIFICATION_METRICS:
        check(probability(metrics.get(key)), f'{label}: missing/invalid {key}')
    matrix = metrics.get('confusion_matrix', [])
    shape_ok = len(matrix) == 6 and all(isinstance(row, list) and len(row) == 6 for row in matrix)
    check(shape_ok, f'{label}: confusion matrix must be 6x6')
    if shape_ok:
        check(sum(sum(row) for row in matrix) == 270, f'{label}: confusion matrix must cover 270 samples')


def history_complete(path, epochs):
    rows = read_json(path)
    if isinstance(rows, list):
        check([row.get('epoch') for row in rows] == list(range(1, epochs + 1)),
              f'{path}: expected all {epochs} epochs in order')
    elif rows is not None:
        check(False, f'{path}: history must be an array')


def check_pdf(path):
    if not require_file(path):
        return
    try:
        document = PdfReader(path)
        check(len(document.pages) > 0, f'{path}: PDF has no pages')
        check(any((page.extract_text() or '').strip() for page in document.pages),
              f'{path}: PDF has no extractable report text')
        cover_text = document.pages[0].extract_text() or ''
        check('Team07' in cover_text and 'TeamXX' not in cover_text, f'{path}: cover team mismatch')
        check('姓名' in cover_text and '學號' in cover_text, f'{path}: missing private completion fields')
        report_text = ''.join(page.extract_text() or '' for page in document.pages)
        for phrase in ('AI協助', 'AI 協助', '自行填', '待使用者', '學生應', '未代為上傳'):
            check(phrase not in report_text, f'{path}: obsolete editorial notice: {phrase}')
        counts['pdf_files'] += 1
    except Exception as error:
        check(False, f'Unreadable PDF {path}: {error}')


def check_references(value, stage_root, json_path, key=''):
    if isinstance(value, dict):
        for child_key, child in value.items():
            check_references(child, stage_root, json_path, child_key)
    elif isinstance(value, list):
        for child in value:
            check_references(child, stage_root, json_path, key)
    elif isinstance(value, str) and (key in PATH_KEYS or key == 'visualizations'):
        if value.startswith(('https://', 'http://')):
            return
        check(not PERSONAL_PATH.search(value), f'{json_path}: absolute path in {key}: {value}')
        candidates = [json_path.parent / value, stage_root / value]
        found = next((path for path in candidates if path.is_file()), None)
        if check(found is not None, f'{json_path}: missing {key} reference: {value}'):
            require_file(found)


def stage1(root, config):
    experiments = root / 'experiments'
    for name in ('traditional', 'E1', 'E2', 'E3', 'E4'):
        directory = experiments / name
        metrics = read_json(directory / 'metrics.json')
        if isinstance(metrics, dict):
            classification_metrics(metrics, f'Stage 1 {name}')
            check(metrics.get('train_samples') == 1260 and metrics.get('validation_samples') == 270,
                  f'Stage 1 {name}: wrong split counts')
            if name != 'traditional':
                check(metrics.get('epochs_budget') == 40 and metrics.get('smoke_only') is False,
                      f'Stage 1 {name}: expected completed 40-epoch formal experiment')
        require_file(directory / 'predictions.json')
        require_file(directory / 'confusion_matrix.png')
        if name == 'traditional':
            require_file(directory / 'pca_svm.joblib')
        else:
            history_complete(directory / 'history.json', 40)
            require_file(directory / 'training_curves.png')
            require_file(directory / 'best_model.pth')
    summary = read_json(experiments / 'summary.json')
    if isinstance(summary, dict):
        check(set(summary.get('experiments', {})) == {'traditional', 'E1', 'E2', 'E3', 'E4'},
              'Stage 1: summary must include traditional and E1-E4')
        chosen = summary.get('selected_experiment')
        check(chosen in ('E2', 'E3', 'E4'), 'Stage 1: final model must be an improved CNN')
        check(config.get('selected_experiment') == chosen, 'Stage 1: config/summary selection mismatch')


def stage2(root, config):
    expected = {f'{architecture}__{strategy}' for architecture in ('resnet18', 'mobilenet_v3_small')
                for strategy in ('feature_extraction', 'full_fine_tuning')}
    for name in sorted(expected):
        directory = root / 'experiments' / name
        result = read_json(directory / 'metrics.json')
        if isinstance(result, dict):
            classification_metrics(result.get('metrics'), f'Stage 2 {name}')
            check(result.get('experiment') == name and result.get('epochs') == 15,
                  f'Stage 2 {name}: wrong experiment identity or epoch budget')
            check((result.get('seed'), result.get('input_size'), result.get('batch_size')) == (42, 224, 32),
                  f'Stage 2 {name}: comparison settings must be seed42/input224/batch32')
        history_complete(directory / 'history.json', 15)
        for filename in ('best.pth', 'predictions.json', 'curves.png', 'confusion_matrix.png'):
            require_file(directory / filename)
    summary = read_json(root / 'experiments/summary.json')
    if isinstance(summary, dict):
        results = summary.get('experiments', [])
        check({result.get('experiment') for result in results} == expected and len(results) == 4,
              'Stage 2: summary must contain exactly the full 2x2 comparison')
        check(summary.get('final_experiment') == config.get('experiment'),
              'Stage 2: config/summary selection mismatch')
    evaluation = read_json(root / 'evaluation/metrics.json')
    if isinstance(evaluation, dict):
        classification_metrics(evaluation.get('metrics'), 'Stage 2 independent evaluation')
        check(isinstance(evaluation.get('latency'), dict), 'Stage 2: missing final inference benchmark')
    require_file(root / 'evaluation/confusion_matrix.png')


def stage3(root, config):
    check(config.get('bbox_format') == 'xyxy_pixel', 'Stage 3: bbox_format must be xyxy_pixel')
    check(config.get('training', {}).get('epochs') == 60, 'Stage 3: expected 60-epoch training budget')
    selection = read_json(root / 'experiments/selection.json')
    if isinstance(selection, dict):
        check(selection.get('epochs_completed') == 60, 'Stage 3: training did not complete 60 epochs')
        csv_path = root / selection.get('training_results_csv', 'experiments/train/results.csv')
        if require_file(csv_path):
            with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
            check([int(row['epoch']) for row in rows] == list(range(1, 61)),
                  'Stage 3: results.csv must contain epochs 1-60')
    summary = read_json(root / 'experiments/summary.json')
    if isinstance(summary, dict):
        check(summary.get('validation_images') == 270 and summary.get('validation_instances') == 641,
              'Stage 3: wrong validation image/instance count')
        check(probability(summary.get('official_metrics', {}).get('map50')), 'Stage 3: missing/invalid mAP@0.5')
        for key in ('precision', 'recall'):
            check(probability(summary.get('fixed_threshold_metrics', {}).get('micro', {}).get(key)),
                  f'Stage 3: missing/invalid fixed-threshold {key}')
        figures = summary.get('visualizations', [])
        check(len(set(figures)) >= 10, 'Stage 3: fewer than 10 distinct GT/prediction visualizations')
        failures = summary.get('failure_cases', [])
        check(len(failures) >= 5 and summary.get('failure_case_count') == len(failures),
              'Stage 3: fewer than 5 documented failure cases or count mismatch')
        check(all(case.get('figure') for case in failures), 'Stage 3: failure case is missing a figure reference')
        check(isinstance(summary.get('latency'), dict), 'Stage 3: separate inference benchmark not completed')
    require_file(root / 'experiments/validation_predictions.json')
    require_file(root / 'experiments/threshold_sweep.json')
    export = read_json(root / 'experiments/checkpoint_export.json')
    if isinstance(export, dict):
        check(export.get('model_tensors_equal') is True and export.get('validation_equal') is True,
              'Stage 3: inference export has not passed state/output equivalence checks')
        check(export.get('inference_checkpoint_bytes') == (root / config['checkpoint_file']).stat().st_size,
              'Stage 3: inference checkpoint size evidence is stale')


def deployment_benchmark(root, stage):
    evidence = read_json(root / 'experiments/deployment_benchmark.json')
    if not isinstance(evidence, dict):
        return
    rows = evidence.get('models', [])
    check(len(rows) == (4 if stage == 2 else 1), f'Stage {stage}: incomplete deployment comparison')
    for row in rows:
        label = f"Stage {stage} {row.get('experiment')} benchmark"
        check(row.get('samples') == 270 and len(set(row.get('image_ids', []))) == 270,
              f'{label}: expected 270 distinct validation images')
        check(row.get('class_counts') == {str(i): 45 for i in range(6)}, f'{label}: class coverage mismatch')
        check((row.get('warmup_iterations'), row.get('precision'), row.get('batch_size')) == (20, 'FP32', 1),
              f'{label}: timing conditions mismatch')
        timings = row.get('sample_ms', [])
        check(len(timings) == 270 and all(isinstance(ms, (float, int)) and ms > 0 for ms in timings),
              f'{label}: invalid per-image timing evidence')
        if timings:
            check(abs(sum(timings) / len(timings) - row.get('mean_ms', 0)) < 1e-9,
                  f'{label}: mean does not match raw timings')
    if stage == 3 and len(rows) == 1:
        summary = read_json(root / 'experiments/summary.json')
        latency = read_json(root / 'experiments/latency.json')
        check(summary.get('latency') == latency, 'Stage 3: summary and latency evidence differ')
        check(rows[0].get('sample_ms') == latency.get('sample_ms'),
              'Stage 3: deployment report uses stale timing samples')
        check(evidence['protocol'].get('measured_at_utc') == latency.get('measured_at_utc') == rows[0].get('measured_at_utc'),
              'Stage 3: timing evidence timestamps differ')
        check(evidence['protocol'].get('cpu_threads') == 4 and evidence['protocol'].get('seed') == 42
              and evidence['protocol'].get('cudnn_deterministic') is True,
              'Stage 3: benchmark execution settings are not recorded consistently')


def check_archive(archive, stage_root):
    if not require_file(archive):
        return
    check(archive.stat().st_size <= 1_000_000_000, f'{archive}: exceeds 1 GB')
    try:
        with zipfile.ZipFile(archive) as zipped:
            entries = [entry for entry in zipped.infolist() if not entry.is_dir()]
            names = {entry.filename for entry in entries}
            check(bool(entries), f'{archive}: empty archive')
            check(len(names) == len(entries), f'{archive}: duplicate member names')
            for name in names:
                path = PurePosixPath(name)
                check(not path.is_absolute() and '..' not in path.parts and '\\' not in name,
                      f'{archive}: unsafe/nonportable member: {name}')
                check(len(path.parts) >= 2 and path.parts[0] == stage_root.name,
                      f'{archive}: incorrect top-level root: {name}')
                check(len(path.parts) < 2 or path.parts[1] != stage_root.name,
                      f'{archive}: duplicated nested stage root: {name}')
                check(not (set(path.parts[1:]) & EXCLUDED_PARTS) and path.suffix not in ('.pyc', '.pyo', '.cache')
                      and path.name != 'yolo11n.pt',
                      f'{archive}: forbidden generated/data member: {name}')
            required = sorted(path.relative_to(stage_root).as_posix() for path in required_files
                              if path.is_relative_to(stage_root))
            for relative in required:
                member = f'{stage_root.name}/{relative}'
                if check(member in names, f'{archive}: missing required file/evidence {relative}'):
                    local = stage_root / relative
                    if local.is_file():
                        check(zipped.read(member) == local.read_bytes(), f'{archive}: stale packaged file {relative}')
            check(zipped.testzip() is None, f'{archive}: ZIP CRC failure')
            counts['zip_files'] += 1
    except (OSError, zipfile.BadZipFile) as error:
        check(False, f'Unreadable ZIP {archive}: {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--team', default='Team07')
    args = parser.parse_args()
    project = args.root.resolve()
    for stage in (1, 2, 3):
        root = project / 'submission' / f'{args.team}_Stage{stage}'
        checkpoint = f'stage{stage}_model.' + ('pt' if stage == 3 else 'pth')
        required = [f'stage{stage}_report.pdf', checkpoint, 'config.json', 'requirements.txt', 'README.md',
                    'src/model.py', 'src/train.py', 'src/evaluate.py']
        for relative in required:
            require_file(root / relative)
        config = read_json(root / 'config.json')
        if not isinstance(config, dict):
            config = {}
        check(CONFIG_FIELDS[stage] <= config.keys(), f'Stage {stage}: missing config fields {sorted(CONFIG_FIELDS[stage] - config.keys())}')
        check(config.get('class_order') == CLASS_ORDER, f'Stage {stage}: class_order mismatch')
        check(config.get('stage') == stage and config.get('seed') == 42, f'Stage {stage}: wrong stage or primary seed')
        check(config.get('team_id') == args.team, f'Stage {stage}: inconsistent team_id')
        check(config.get('checkpoint_file') == checkpoint, f'Stage {stage}: wrong final checkpoint_file')
        if config.get('team_id') == 'TeamXX':
            pending_metadata.append('Fill the actual iLearning team number before submission.')
        pending_metadata.append('Cover member fields are blank.')
        for path in [*root.glob('src/*.py'), root / 'config.json', root / 'README.md']:
            if path.is_file():
                require_file(path)
                check(not PERSONAL_PATH.search(path.read_text(encoding='utf-8')),
                      f'Hardcoded personal absolute path: {path}')
        for path in sorted(root.rglob('*.json')):
            if set(path.relative_to(root).parts) & EXCLUDED_PARTS:
                continue
            value = read_json(path)
            if value is not None:
                check_references(value, root, path)
        for path in sorted(root.rglob('*')):
            if path.suffix.lower() not in ('.png', '.jpg', '.jpeg') or set(path.relative_to(root).parts) & EXCLUDED_PARTS:
                continue
            try:
                require_file(path)
                with Image.open(path) as image:
                    image.verify()
                counts['image_files'] += 1
            except Exception as error:
                check(False, f'Unreadable image {path}: {error}')
        {1: stage1, 2: stage2, 3: stage3}[stage](root, config)
        deployment_benchmark(root, stage)
        check_pdf(root / f'stage{stage}_report.pdf')
        check_archive(project / 'output' / f'{root.name}.zip', root)
    check_pdf(project / 'output' / f'{args.team}_Final_Report.pdf')
    result = {'status': 'FAIL' if errors else 'PASS', 'checks': counts,
              'errors': list(dict.fromkeys(errors)), 'pending_metadata': sorted(set(pending_metadata)),
              'scope': 'Read-only artifact/format/evidence checks. Weight reload and visual PDF review are separate checks.'}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
