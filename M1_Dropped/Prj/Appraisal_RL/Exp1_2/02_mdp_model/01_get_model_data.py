from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from improved_baseline.pipeline import run_experiment


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[1] / "data"
    result_files, _ = run_experiment("exp12", output_dir=output_dir)
    print(f"Wrote {result_files[0]}")
    print(f"Wrote {result_files[1]}")
