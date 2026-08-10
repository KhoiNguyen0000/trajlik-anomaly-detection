"""Validate a trajectory cache produced by src.cache_trajectories."""

import argparse
import json
from pathlib import Path

import torch


REQUIRED_TENSORS = ("z_0", "z_seq", "eps_seq", "delta_z_seq")
DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument(
        "--expected_categories",
        nargs="*",
        default=None,
        help="Optional list of categories that must appear in the cache.",
    )
    return parser.parse_args()


def validate_cache(cache_dir: Path, expected_categories=None):
    metadata_path = cache_dir / "cache_meta.json"
    assert metadata_path.is_file(), f"Missing metadata: {metadata_path}"

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    num_steps = int(metadata["num_steps"])
    channels = int(metadata["output_channels"])
    storage_dtype = metadata["storage_dtype"]
    assert storage_dtype in DTYPES, f"Unsupported dtype: {storage_dtype}"
    expected_dtype = DTYPES[storage_dtype]

    projection = metadata["projection"]
    timestep_map = metadata.get("timestep_map")
    if timestep_map is not None:
        assert len(timestep_map) == num_steps, (
            f"Timestep map has {len(timestep_map)} entries, expected {num_steps}"
        )
        assert timestep_map == sorted(timestep_map), (
            f"Timestep map must be sorted, got {timestep_map}"
        )

    projector_path = cache_dir / "projector.pt"
    if projection == "linear":
        assert projector_path.is_file(), "Linear projection requires projector.pt"

    cache_files = sorted(
        path
        for path in cache_dir.rglob("*.pt")
        if path.name != "projector.pt"
    )
    assert cache_files, f"No cached image files found in {cache_dir}"
    assert len(cache_files) == int(metadata["num_images"]), (
        f"Metadata declares {metadata['num_images']} images, "
        f"but {len(cache_files)} files were found"
    )

    observed_categories = set()
    for path in cache_files:
        sample = torch.load(path, map_location="cpu", weights_only=False)

        missing = set(REQUIRED_TENSORS) - sample.keys()
        assert not missing, f"{path}: missing keys {sorted(missing)}"

        z_0 = sample["z_0"]
        expected_z_shape = (channels, z_0.shape[-2], z_0.shape[-1])
        expected_seq_shape = (num_steps, *expected_z_shape)

        assert tuple(z_0.shape) == expected_z_shape, (
            f"{path}: z_0 has shape {tuple(z_0.shape)}, "
            f"expected {expected_z_shape}"
        )
        assert z_0.dtype == expected_dtype, (
            f"{path}: z_0 has dtype {z_0.dtype}, expected {expected_dtype}"
        )

        for key in REQUIRED_TENSORS[1:]:
            value = sample[key]
            assert tuple(value.shape) == expected_seq_shape, (
                f"{path}: {key} has shape {tuple(value.shape)}, "
                f"expected {expected_seq_shape}"
            )
            assert value.dtype == expected_dtype, (
                f"{path}: {key} has dtype {value.dtype}, expected {expected_dtype}"
            )

        category = sample.get("category")
        assert category, f"{path}: missing category"
        observed_categories.add(category)

        source_path = str(sample.get("source_path", "")).replace("\\", "/").lower()
        assert "/train/" in source_path, f"{path}: source is not from the training split"
        assert "/good/" in source_path, f"{path}: source is not a normal image"

    if expected_categories:
        missing_categories = set(expected_categories) - observed_categories
        assert not missing_categories, (
            f"Missing categories: {sorted(missing_categories)}"
        )

    print(
        f"Passed: {len(cache_files)} files, "
        f"{len(observed_categories)} categories, "
        f"{num_steps} steps, {channels} channels, {storage_dtype}"
    )


if __name__ == "__main__":
    args = parse_args()
    validate_cache(args.cache_dir, args.expected_categories)
