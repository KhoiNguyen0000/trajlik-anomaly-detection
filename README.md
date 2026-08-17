# TrajLik-AD

Trajectory-likelihood anomaly detection built on a frozen InvAD baseline. The
project adds DCTE trajectory encoding, ECTF likelihood estimation, normal-only
calibration, and endpoint/path score fusion.

## Structure

- `dcte/`, `ectf/`: core TrajLik modules.
- `scripts/`: cache, train, evaluate, and reproducibility entrypoints.
- `tests/`: unit, integration, and baseline regression tests.
- `baseline/InversionAD/`: InvAD baseline implementation and configs.
- `docs/`: cache guide and project documents.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m scripts.cache_trajectories --help
python -m scripts.train_trajlik --help
python -m scripts.evaluate_trajlik --help
```

Run all tests:

```bash
python -m unittest discover -s tests -t . -p "test_*.py" -v
```

See [TRAJLIK.md](TRAJLIK.md) for the pipeline and
[docs/CACHE_TRAJECTORIES.md](docs/CACHE_TRAJECTORIES.md) for cache generation.
