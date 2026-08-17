import argparse
from pathlib import Path

from trajlik.cache_layout import organize_cache


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group trajectory cache files into category directories",
    )
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move files and update cache_index.json; default is a dry run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every planned move instead of only the first 20",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plan = organize_cache(args.cache_dir, apply=args.apply)
    visible = plan if args.verbose else plan[:20]
    for source, destination in visible:
        print(f"{source.name} -> {destination.parent.name}/{destination.name}")
    if len(visible) < len(plan):
        print(f"... and {len(plan) - len(visible)} more files")
    action = "Moved" if args.apply else "Would move"
    print(f"{action} {len(plan)} cache files")
    if plan and not args.apply:
        print("Dry run only; rerun with --apply to make these changes")


if __name__ == "__main__":
    main()
