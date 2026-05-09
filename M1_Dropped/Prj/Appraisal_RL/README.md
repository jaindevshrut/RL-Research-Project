# Appraisal_RL
Please refer to the readme.md in folders.

## Improved baseline-compatible pipeline

Install the shared dependencies with `pip install -r requirements.txt`, then run:

- `python run_improved_baseline.py --experiment all`

This keeps the original folder structure and writes the standard `data/model_result.csv`
and `data/svm_*.csv` files for `Exp1_2` and `Exp3`, while also adding:

- `data/model_result_6d.csv`
- `data/mlp_ensemble_predictions.csv`
