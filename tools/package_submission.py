"""Package standalone Stage deliverables, excluding local caches and training data."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package_stage(number):
    name = f'Team07_Stage{number}'
    source = ROOT / 'submission' / name
    selected_weight = None
    if number == 3:
        selection = json.loads((source / 'experiments/selection.json').read_text(encoding='utf-8'))
        selected_weight = selection['source_checkpoint']
    destination = ROOT / 'output' / f'{name}.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(source.rglob('*')):
            if not file.is_file():
                continue
            relative = file.relative_to(source)
            if any(part in ('__pycache__', 'smoke', 'prepared_data') for part in relative.parts):
                continue
            if file.suffix == '.cache' or relative.name == 'yolo11n.pt':
                continue
            if number == 3 and 'weights' in relative.parts and relative.as_posix() != selected_weight:
                continue
            archive.write(file, (Path(name) / relative).as_posix())
    print(f'{destination.name}: {destination.stat().st_size / 1024**2:.2f} MiB')


def main():
    (ROOT / 'output').mkdir(exist_ok=True)
    for number in (1, 2, 3):
        package_stage(number)


if __name__ == '__main__':
    main()
