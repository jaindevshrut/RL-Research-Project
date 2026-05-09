# Improved Appraisal-RL Model

An improved implementation of the paper *"Modeling Cognitive-Affective Processes with Appraisal and Reinforcement Learning"* (arXiv:2309.06367v2) with **3 novel contributions**.

## Novel Improvements

1. **DQN Agent** — Deep Q-Network with experience replay and target network replaces tabular Q-learning
2. **Extended 6D Appraisal** — Two new CPM checks: Intrinsic Unpredictability (transition entropy) and Normative Significance (TD error z-score)
3. **Neural Classifier Ensemble** — 10 MLP classifiers with dropout replace the single SVM

## Setup

```bash
cd improved
pip install -r requirements.txt
```

> **Note:** PyTorch may need to be installed separately depending on your system. See [pytorch.org](https://pytorch.org/get-started/locally/).

## Run

```bash
python improved_experiment.py
```

This will:
1. Train DQN agents on all 11 emotion scenarios (5 runs each for stability)
2. Compute 6-dimensional appraisal vectors
3. Train neural classifier ensemble + SVM baseline
4. Print a comparison table vs the original paper results
5. Save results to `results/` directory

## Project Structure

```
improved/
├── environments.py          # 11 unified MDP environments (AppraisalMDP base class)
├── dqn_agent.py             # DQN agent with experience replay
├── appraisal.py             # 6D appraisal vector computation
├── neural_classifier.py     # MLP ensemble + SVM baseline
├── improved_experiment.py   # Main experiment runner
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── results/                 # Output directory (created at runtime)
    ├── model_result_6d.csv
    ├── comparison_summary.csv
    └── emotion_predictions.csv
```

## Output

The script prints:
- **Appraisal vectors** for all 11 emotions (6D)
- **Emotion predictions** with confidence scores
- **Comparison table** showing R², RMSE, accuracy vs original paper
- **Novel contributions summary**
