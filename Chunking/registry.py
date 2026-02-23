import json
from pathlib import Path


def load_registry(path: Path) -> set:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def save_registry(path: Path, registry: set):
    path.write_text(
        json.dumps(sorted(registry), indent=4),
        encoding="utf-8"
    )
