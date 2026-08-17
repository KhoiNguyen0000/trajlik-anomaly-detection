import json
import re
from pathlib import Path


def sanitize_category(category: str) -> str:
    """Return a filesystem-safe category directory name."""

    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(category)).strip("._")
    if not value:
        raise ValueError(f"Invalid cache category: {category!r}")
    return value


def _resolve_indexed_path(cache_dir: Path, relative_path: str) -> Path:
    path = (cache_dir / relative_path).resolve()
    if path != cache_dir and cache_dir not in path.parents:
        raise ValueError(f"Cache index path escapes cache directory: {relative_path}")
    return path


def build_organization_plan(cache_dir):
    """Plan category-directory moves and the corresponding updated index."""

    cache_dir = Path(cache_dir).resolve()
    index_path = cache_dir / "cache_index.json"
    with index_path.open(encoding="utf-8") as file:
        index = json.load(file)
    if not isinstance(index, list):
        raise ValueError("cache_index.json must contain a list")

    plan = []
    updated_index = []
    destinations = set()
    for entry in index:
        relative_path = entry.get("file")
        category = entry.get("category")
        if not relative_path or not category:
            raise ValueError("Every cache index entry needs file and category")

        source = _resolve_indexed_path(cache_dir, relative_path)
        if not source.is_file():
            raise FileNotFoundError(f"Indexed cache sample is missing: {source}")
        destination = cache_dir / sanitize_category(category) / source.name
        destination = destination.resolve()
        if destination in destinations:
            raise FileExistsError(f"Duplicate organized path: {destination}")
        if destination != source and destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        destinations.add(destination)

        updated_entry = dict(entry)
        updated_entry["file"] = destination.relative_to(cache_dir).as_posix()
        updated_index.append(updated_entry)
        if destination != source:
            plan.append((source, destination))

    return plan, updated_index


def organize_cache(cache_dir, *, apply=False):
    """Group indexed cache tensors by category and update the index atomically."""

    cache_dir = Path(cache_dir).resolve()
    plan, updated_index = build_organization_plan(cache_dir)
    if not apply:
        return plan

    moved = []
    temporary_index = cache_dir / "cache_index.json.tmp"
    try:
        for source, destination in plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            moved.append((source, destination))

        temporary_index.write_text(
            json.dumps(updated_index, indent=2),
            encoding="utf-8",
        )
        temporary_index.replace(cache_dir / "cache_index.json")
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(source)
        temporary_index.unlink(missing_ok=True)
        raise

    for directory in sorted(cache_dir.rglob("*"), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass
    return plan
