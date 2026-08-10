import importlib.metadata
import json
import subprocess
from pathlib import Path


def validate_main_protocol(config):
    """Return all config mismatches against the locked TrajLik main protocol."""

    errors = []

    def require(actual, expected, path):
        if actual != expected:
            errors.append(f"{path}: expected {expected!r}, got {actual!r}")

    data = config.get("data", {})
    backbone = config.get("backbone", {})
    diffusion = config.get("diffusion", {})
    evaluation = config.get("evaluation", {})
    require(data.get("img_size"), 256, "data.img_size")
    require(data.get("transform_type"), "imagenet", "data.transform_type")
    require(backbone.get("model_type"), "efficientnet-b4", "backbone.model_type")
    require(backbone.get("outblocks"), [1, 5, 9, 21], "backbone.outblocks")
    require(backbone.get("stride"), 16, "backbone.stride")
    require(diffusion.get("model_type"), "dit", "diffusion.model_type")
    require(evaluation.get("eval_step"), 3, "evaluation.eval_step")

    dataset_name = str(data.get("dataset_name", ""))
    data_root = str(data.get("data_root", "")).replace("\\", "/").lower()
    expected_roots = {
        "mvtec_ad": "mvtec_ad",
        "mvtec_ad_all": "mvtec_ad",
        "visa": "visa",
        "visa_all": "visa",
        "mpdd": "mpdd",
        "mpdd_all": "mpdd",
    }
    expected_root = expected_roots.get(dataset_name)
    if expected_root is None:
        errors.append(f"data.dataset_name: unsupported main dataset {dataset_name!r}")
    elif expected_root not in data_root:
        errors.append(
            f"data.data_root: {data_root!r} does not match {dataset_name!r}"
        )
    return errors


def pinned_version_report(requirements_path):
    mismatches = []
    checked = {}
    for raw_line in Path(requirements_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        package, expected = line.split("==", 1)
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        checked[package] = {"expected": expected, "actual": actual}
        if actual != expected:
            mismatches.append(
                f"package {package}: expected {expected}, got {actual or 'not installed'}"
            )
    return checked, mismatches


def git_state(repo_dir):
    repo_dir = Path(repo_dir)

    def run(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--short")),
    }


def build_reproducibility_report(
    config,
    config_path,
    requirements_path,
    repo_dir,
    checkpoint_path=None,
    check_dataset=True,
):
    errors = validate_main_protocol(config)
    versions, version_errors = pinned_version_report(requirements_path)
    errors.extend(version_errors)

    data_root = Path(config["data"]["data_root"])
    if not data_root.is_absolute():
        data_root = Path(repo_dir) / data_root
    if check_dataset and not data_root.exists():
        errors.append(f"dataset root does not exist: {data_root.resolve()}")

    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = str(Path(checkpoint_path).resolve())
        if not Path(checkpoint_path).is_file():
            errors.append(f"checkpoint does not exist: {checkpoint}")

    return {
        "valid": not errors,
        "errors": errors,
        "config_path": str(Path(config_path).resolve()),
        "checkpoint_path": checkpoint,
        "dataset_root": str(data_root.resolve()),
        "git": git_state(repo_dir),
        "packages": versions,
    }


def write_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
