import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_DIR = Path(__file__).parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.reproducibility import build_reproducibility_report, write_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit InvAD/TrajLik config, environment, data, and checkpoint",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="reproducibility_report.json")
    parser.add_argument(
        "--skip_dataset",
        action="store_true",
        help="Do not require the configured dataset root to exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_dir = REPO_DIR
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    report = build_reproducibility_report(
        config=config,
        config_path=args.config,
        requirements_path=repo_dir / "requirements.txt",
        repo_dir=repo_dir,
        checkpoint_path=args.checkpoint,
        check_dataset=not args.skip_dataset,
    )
    write_report(report, args.output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
