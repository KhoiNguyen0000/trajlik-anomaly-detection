import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = PROJECT_ROOT / "baseline" / "InversionAD"
BACKBONE_ROOT = BASELINE_ROOT / "src" / "backbones"


for path in (PROJECT_ROOT, BASELINE_ROOT, BACKBONE_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
