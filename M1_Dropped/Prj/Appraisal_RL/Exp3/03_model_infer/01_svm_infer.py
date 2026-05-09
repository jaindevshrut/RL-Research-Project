from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from improved_baseline.pipeline import run_classifier_suite


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    outputs = run_classifier_suite("exp3", data_dir=data_dir)
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")
