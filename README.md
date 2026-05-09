# Modeling Cognitive-Affective Processes With Appraisal and Reinforcement Learning

A B.Tech research project at SVNIT, Surat. We extend Zhang, Broekens
and Jokinen (arXiv:2309.06367, 2023), who mapped four appraisal checks
from Scherer's Component Process Model (CPM) onto a tabular Q-learning
agent. The paper documents **four methods** in total along two design
tracks. The two retained configurations are compared head-to-head on
the same eight-dimensional appraisal vector:

- **Method 1b** (the headline configuration): a Dueling Double DQN
  that carries a bootstrap ensemble of K=3 value heads.
- **Method 2b** (a controlled comparison): a Quantile-Regression DQN
  with N=51 quantile atoms, following Dabney *et al.*, AAAI 2018.

Method 1b leads on every appraisal-derived metric and is the
configuration reported in the paper's title and abstract. Method 2b
is included as a transparent, runnable diagnostic that isolates
whether Method 1b's gain is specific to its representation. (It is.)

Two **preliminary configurations** along the same design tracks are
documented in the paper's methodology with the empirical evidence
that motivated their replacement; their code lives outside the
reproducible pipeline:

- **Method 1a** (preliminary, dropped — `M1_Dropped/`): a vanilla
  Deep Q-Network paired with a six-dimensional appraisal vector
  (the four Scherer channels plus a transition-entropy
  *intrinsic unpredictability* and a TD-error z-score
  *normative significance*) and a ten-head MLP ensemble classifier.
  Dropped because the MLP ensemble offered no consistent advantage
  over a single calibrated SVM and the six-dimensional vector left
  the Exp3_Free condition with a near-zero R² (0.021 / 0.034) and
  top-1 accuracy of only 4/7 on Exp1_2 — far below the
  eight-dimensional Method 1b that replaces it.
- **Method 2a** (preliminary, dropped — `M2_Dropped/`): a
  64-quantile Distributional RL value head paired with a calibrated
  linear SVM on a four-dimensional Scherer mapping
  (Goal_relevance ← Var(Z), Conduciveness ← E[Z], Power ← action-value
  variance, Fear ← negative-mass fraction). Dropped because the
  64-quantile distribution was collapsed to those scalar summaries
  before reaching the SVM, so the appraisal head consumed strictly
  less information than the value head learned. Aggregate R² across
  five seeds collapsed to **−0.326** against a tabular four-dimensional
  baseline of 0.501, with aggregate RMSE 40.7% worse. Method 2b
  fixes the structural deficit by sourcing the predictability
  channel from intra-distribution spread directly, on the same
  eight-dimensional vector that Method 1b uses.

## Authors

| Name              | Enrollment | Email                          | Role              |
|-------------------|------------|--------------------------------|-------------------|
| Dev Shrut Jain    | U23AI068   | jaindevshrut@gmail.com         | Team Leader       |
| Sweta Rana        | U23AI065   | ranasweta2005@gmail.com        | Author            |
| Krish Rathod      | U23AI049   | krishrathod5696@gmail.com      | Author            |

See `AUTHOR_CONTRIBUTIONS.txt` for a per-author breakdown of who did
what.

## Headline result

| Exp.  | Configuration | Backbone  | Dim. | Acc.  | R²    | RMSE  |
|-------|---------------|-----------|------|-------|-------|-------|
| Exp 1 | Baseline-4D   | Method 1b | 4    | 0.648 | 0.366 | 0.345 |
| Exp 2 | Extended-8D   | Method 1b | 8    | **0.903** | **0.757** | **0.214** |
| Exp 3 | Extended-8D   | Method 2b | 8    | 0.868 | 0.694 | 0.239 |

Method 1a and Method 2a are not in this table because they were
evaluated on the Scherer-style 7-emotion / 4-emotion protocol rather
than the four task-event proxies (neutral, joy, satisfaction,
despair) used here; their per-experiment numbers are reported inline
in the corresponding methodology subsections of the paper.

Identical seed, identical environment, identical 60,000-frame budget
across all three runs. The eight-dimensional appraisal vector lifts
classification accuracy by **+25.5 percentage points** over the
4-dimensional paper-faithful baseline, and the gain is concentrated in
the four newly added channels (familiarity, power, anticipation,
predictability — confirmed by a permutation-importance analysis).

## Repository layout

```
RL-Research-Project/
├── README.md                        # this file (project overview, reproduce)
├── AUTHOR_CONTRIBUTIONS.txt         # who did what
├── requirements.txt                 # numpy, torch, matplotlib (CPU only)
├── .gitignore
│
├── paper/                           # publishable artefacts
│   ├── conference_101719.tex        # IEEE-conference LaTeX source
│   ├── conference_101719.pdf        # compiled paper (13 pages)
│   └── figures/                     # all figures used by the .tex
│
├── presentation/                    # slide deck for the project review
│   └── RL-Research-Project.pptx
│
├── M1_Dropped/                      # Method 1a (preliminary, dropped)
│   └── Prj/Appraisal_RL/            # DQN + 6D appraisal + 10-MLP ensemble
│                                    # documented in the paper §IV.C
├── M2_Dropped/                      # Method 2a (preliminary, dropped)
│   ├── Exp1_2/                      # 64-quantile Distributional RL +
│   └── Exp3/                        # calibrated SVM, documented §IV.E
│
├── src/                             # executable code (Method 1b + 2b)
│   ├── config.py                    # all hyperparameters in one dataclass
│   ├── env/gridworld.py             # 7×7 key-and-lava grid world
│   ├── replay_buffer.py             # uniform + prioritised replay
│   ├── networks/
│   │   ├── dueling_dqn.py           # Method 1b: dueling + ensemble
│   │   └── qr_dqn.py                # Method 2b: quantile-regression head
│   ├── appraisal/extractor.py       # 8-dim appraisal extraction (shared)
│   ├── agent.py                     # Method 1b agent
│   ├── agent_qrdqn.py               # Method 2b agent
│   ├── train.py                     # Method 1b training entry-point
│   └── train_qrdqn.py               # Method 2b training entry-point
│
├── analysis/                        # everything used to produce paper plots
│   ├── correlation_analysis.py      # Pearson, VIF, effective rank
│   ├── explainability.py            # permutation importance + emotion map
│   ├── rmse_r2.py                   # master-table R² / RMSE / accuracy
│   ├── compare_runs.py              # side-by-side summary
│   └── generate_figures.py          # regenerates every paper figure
│
├── runs/                            # output artefacts, one folder per run
│   ├── baseline_4dim/               # Exp 1 (paper-faithful 4-D)
│   │   ├── appraisals.npz
│   │   ├── returns.npy
│   │   ├── eval.json
│   │   └── config.json
│   ├── extended_8dim/               # Exp 2 (Method 1b, headline)
│   │   └── ...
│   ├── qrdqn_8dim/                  # Exp 3 (Method 2b, comparison)
│   │   └── ...
│   └── rmse_r2_summary.json         # consolidated metrics table
│
└── docs/                            # supporting write-ups
    ├── PROJECT_REPORT.md
    ├── design_rationale.md
    ├── decorrelation_proof.md
    └── results_snapshot.md
```

## Reproducing every number in the paper

CPU-only install. Each training run takes about 15-30 minutes on a
single CPU core; the analysis scripts run in seconds.

### Step 1 — install

```bash
pip install -r requirements.txt
```

### Step 2 — train all three conditions

```bash
# Exp 1 — Baseline-4D (paper-faithful restriction, on Method 1b backbone)
python -m src.train --run baseline_4dim

# Exp 2 — Extended-8D, Method 1b (the headline configuration)
python -m src.train --run extended_8dim

# Exp 3 — Extended-8D, Method 2b (QR-DQN comparison)
python -m src.train_qrdqn --run qrdqn_8dim
```

Each command writes to `runs/<name>/{appraisals.npz, returns.npy,
eval.json, config.json}`.

### Step 3 — compute the master conclusion table

```bash
python -m analysis.rmse_r2 \
    --runs baseline_4dim extended_8dim qrdqn_8dim
```

Writes the consolidated metrics to `runs/rmse_r2_summary.json` and
prints the side-by-side comparison the paper reports in Table VI.

### Step 4 — regenerate every figure in the paper

```bash
python -m analysis.generate_figures
```

Reads from `runs/` and writes to `paper/figures/`. After this step the
PDF can be recompiled from the `paper/` folder with:

```bash
cd paper
pdflatex conference_101719.tex
pdflatex conference_101719.tex
```

(Two passes are needed for cross-references.)

### Step 5 — full decorrelation report (optional)

```bash
python -m analysis.correlation_analysis --run extended_8dim
python -m analysis.explainability --run extended_8dim
```

## How the paper-side artefacts and the code are linked

| Paper element                         | Source artefact                                |
|---------------------------------------|------------------------------------------------|
| Master conclusion table (Table VI)    | `runs/rmse_r2_summary.json`                    |
| Headline comparison (Table V)         | `runs/rmse_r2_summary.json`                    |
| Per-event fingerprint (Table V)       | `runs/extended_8dim/appraisals.npz`            |
| Per-dimension VIF (Table IV)          | `runs/extended_8dim/analysis/report.json`      |
| Permutation importance (Fig. 5)       | `runs/extended_8dim/analysis/explainability.json` |
| Eigenvalue spectrum (Fig. 4)          | derived in-script from `appraisals.npz`        |
| Correlation heatmap (Fig. 3)          | derived in-script from `appraisals.npz`        |
| Comparison bars (Fig. 7)              | `runs/rmse_r2_summary.json`                    |
| Per-class calibration (Fig. 8)        | re-trains classifier in-script                 |
| Confusion matrices (Fig. 9)           | re-trains classifier in-script                 |
| Training curve (Fig. 1)               | parsed from `runs/extended_8dim_log.txt`       |

## Determinism

- `seed = 0` for both NumPy and PyTorch in every training script.
- The grid-world layout, the agent's initial weights and the
  replay-buffer ordering are all deterministic given the seed.
- Numbers in the README and in the paper match `runs/rmse_r2_summary.json`
  exactly; minor floating-point variation (~1e-6) can occur across CUDA
  vs CPU but the paper was produced on CPU.

## Citing the prior work this paper builds on

```
Zhang, Broekens, Jokinen. "Modeling cognitive-affective processes
with appraisal and reinforcement learning." arXiv:2309.06367, 2023.

Dabney, Rowland, Bellemare, Munos. "Distributional reinforcement
learning with quantile regression." AAAI 2018, pp. 2892-2901.
```

## License and contact

This is a B.Tech.\ research project; please reach the team leader or
either author by email (above) for questions or to request the raw
training logs.
