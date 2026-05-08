"""Run the appraisal MDPs across fixed seeds and aggregate model features."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CURRENT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from config import MODEL_TYPE, SEEDS


RESULT_COLUMNS = [
    "Emotion",
    "Suddenness",
    "Goal_relevance",
    "Conduciveness",
    "Power",
]
EMOTION_ORDER = ["Anxiety", "Despair", "Irritation", "Rage"]
PROGRAM_LIST = [
    "02_mdp_model/anxiety.py",
    "02_mdp_model/despair.py",
    "02_mdp_model/irritation.py",
    "02_mdp_model/rage.py",
]
DATA_DIR = EXPERIMENT_DIR / "data"
RESULT_FILE = DATA_DIR / "model_result.csv"
SEED_RUNNER = (
    "import os, random, runpy, sys\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "seed = int(os.environ['SEED'])\n"
    "random.seed(seed)\n"
    "np.random.seed(seed)\n"
    "sys.path.insert(0, str(Path.cwd() / '02_mdp_model'))\n"
    "runpy.run_path(sys.argv[1], run_name='__main__')\n"
)


def reset_result_file() -> None:
    with RESULT_FILE.open("w", newline="") as new_file:
        writer = csv.writer(new_file)
        writer.writerow(RESULT_COLUMNS)


def run_program(program: str, seed: int) -> None:
    env = os.environ.copy()
    env["MODEL_TYPE"] = MODEL_TYPE
    env["SEED"] = str(seed)
    subprocess.check_call(
        [sys.executable, "-c", SEED_RUNNER, program],
        cwd=str(EXPERIMENT_DIR),
        env=env,
    )


def aggregate_seed_results(seed_files):
    frames = []
    for seed_file in seed_files:
        seed = int(seed_file.stem.split("seed")[-1])
        frame = pd.read_csv(seed_file)
        frame["Seed"] = seed
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["Emotion"] = pd.Categorical(
        combined["Emotion"],
        categories=EMOTION_ORDER,
        ordered=True,
    )
    combined = combined.sort_values(["Seed", "Emotion"]).reset_index(drop=True)
    combined.to_csv(DATA_DIR / f"model_result_{MODEL_TYPE}_all_seeds.csv", index=False)

    aggregate = (
        combined.groupby("Emotion", as_index=False)[RESULT_COLUMNS[1:]]
        .mean()
        .sort_values("Emotion")
        .reset_index(drop=True)
    )
    aggregate.to_csv(DATA_DIR / f"model_result_{MODEL_TYPE}.csv", index=False)
    aggregate.to_csv(RESULT_FILE, index=False)


def main() -> None:
    print(f"Running Exp3 model extraction for MODEL_TYPE={MODEL_TYPE!r}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seed_files = []
    for seed in SEEDS:
        print(f"Seed {seed}")
        reset_result_file()
        for program in PROGRAM_LIST:
            run_program(program, seed)
            print(f"Finished: {program}")

        seed_file = DATA_DIR / f"model_result_{MODEL_TYPE}_seed{seed}.csv"
        shutil.copyfile(RESULT_FILE, seed_file)
        seed_files.append(seed_file)

    aggregate_seed_results(seed_files)
    print(f"Saved aggregate features to {RESULT_FILE}")


if __name__ == "__main__":
    main()
