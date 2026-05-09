# Corrected Results — Method 1: DQN + 6D Appraisal

## Top-1 Emotion Classification Accuracy (Actual Values Only)

| Experiment | Accuracy | Correct/Total |
|-----------|----------|---------------|
| **Exp1_2** | 57.14% | 4/7 |
| **Exp3** | 75.00% | 3/4 |

### Exp1_2 Predictions (7 emotions):
- ✓ Boredom → Boredom (1.0000)
- ✓ Fear → Fear (0.9999)
- ✓ Happiness → Happiness (0.9879)
- ✓ Joy → Joy (0.9145)
- ✗ Pride → Joy (1.0000)
- ✗ Sadness → Shame (0.9438)
- ✗ Shame → Fear (0.9998)

### Exp3 Predictions (4 emotions):
- ✓ Anxiety → Anxiety (1.0000)
- ✓ Despair → Despair (1.0000)
- ✗ Irritation → Anxiety (0.9998)
- ✓ Rage → Rage (0.9997)

---

## Appraisal Dimension Performance (R²)

| Metric | Value |
|--------|-------|
| **Exp1_2 Appraisal R²** | 0.8954 |
| **Exp3 Appraisal R²** | 0.8579 |

**Improvement over Baseline:**
- Exp1_2: +0.245 (baseline: 0.65)
- Exp3: +0.238 (baseline: 0.62)

---

## 6D Appraisal Values (Improved Model)

### Emotion Profiles

| Emotion | Suddenness | Goal_relevance | Conduciveness | Power | Intrinsic_unpredictability | Normative_significance |
|---------|-----------|----------------|---------------|-------|---------------------------|----------------------|
| Boredom | 0.0 | ~0 | 0.50 | 0.60 | 0.0 | ~0 |
| Fear | 0.80 | 1.0 | 0.0 | 0.0 | 0.72 | 0.92 |
| Happiness | 0.0 | 0.43 | 0.71 | 0.99 | 0.0 | 1.0 |
| Joy | 0.80 | 1.0 | 1.0 | 0.0 | 0.72 | 0.73 |
| Pride | 0.50 | 0.50 | 0.75 | 0.32 | 1.0 | 0.95 |
| Sadness | 0.20 | 0.74 | 0.13 | 0.0 | 0.72 | 0.49 |
| Shame | 0.79 | 1.0 | 0.0 | 0.38 | 0.74 | 1.0 |
| Anxiety | 0.20 | 0.65 | 0.18 | 0.0 | 0.72 | 0.27 |
| Despair | 0.80 | 1.0 | 0.0 | 0.0 | 0.72 | 0.92 |
| Irritation | 0.20 | 0.89 | 0.05 | 0.04 | 0.72 | 0.22 |
| Rage | 0.80 | 1.0 | 0.0 | 0.71 | 0.72 | 1.0 |

---

## Key Findings

- Only **Exp3 values** from the original graph were accurate (75.0%)
- Exp1_2 actual performance is **57.14%** (not 71.4% as claimed)
- Improved model achieves strong **appraisal R² > 0.85** for both experiments
- All high-confidence emotions (Fear, Joy, Despair, Rage, Shame) show **0.72-1.0 unpredictability values**
