import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load data from improved folder
base_path = Path("improved/results")
emotion_pred_df = pd.read_csv(base_path / "emotion_predictions.csv")
model_result_df = pd.read_csv(base_path / "model_result_6d.csv")
comparison_df = pd.read_csv(base_path / "comparison_summary.csv")

# ==============================================================================
# FIGURE 1: Top-1 Accuracy by Experiment (CORRECTED)
# ==============================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Calculate accuracy by experiment
exp1_2_correct = emotion_pred_df[emotion_pred_df['Experiment'] == 'Exp1_2']['Correct'].sum()
exp1_2_total = len(emotion_pred_df[emotion_pred_df['Experiment'] == 'Exp1_2'])
exp1_2_acc = (exp1_2_correct / exp1_2_total) * 100

exp3_correct = emotion_pred_df[emotion_pred_df['Experiment'] == 'Exp3']['Correct'].sum()
exp3_total = len(emotion_pred_df[emotion_pred_df['Experiment'] == 'Exp3'])
exp3_acc = (exp3_correct / exp3_total) * 100

# Bar chart
experiments = ['Exp1_2', 'Exp3']
accuracies = [exp1_2_acc, exp3_acc]
colors = ['#1f77b4', '#ff7f0e']

bars = ax1.bar(experiments, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax1.set_title('Method 1: DQN + 6D Appraisal\nEmotion Classification Accuracy', 
              fontsize=13, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2f}%\n({int(height*10)}/10)',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# Prediction breakdown table
ax2.axis('off')
table_data = []
for exp in experiments:
    exp_data = emotion_pred_df[emotion_pred_df['Experiment'] == exp]
    correct = exp_data['Correct'].sum()
    total = len(exp_data)
    acc = (correct/total)*100
    table_data.append([exp, correct, total, f'{acc:.2f}%'])

table = ax2.table(cellText=table_data,
                 colLabels=['Experiment', 'Correct', 'Total', 'Accuracy'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0.3, 1, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#1f77b4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(table_data) + 1):
    for j in range(4):
        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

ax2.text(0.5, 0.95, 'Classification Performance Summary', 
         ha='center', fontsize=12, fontweight='bold', transform=ax2.transAxes)

plt.tight_layout()
plt.savefig('m1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: m1_accuracy_comparison.png")
plt.close()

# ==============================================================================
# FIGURE 2: Appraisal R² Scores
# ==============================================================================
fig2, ax = plt.subplots(figsize=(10, 6))

appraisal_metrics = ['Exp1_2 Appraisal R²', 'Exp3 Appraisal R²']
appraisal_r2_values = [
    comparison_df[comparison_df['Metric'] == 'Exp12_Appraisal_R2']['Improved'].values[0],
    comparison_df[comparison_df['Metric'] == 'Exp3_Appraisal_R2']['Improved'].values[0]
]

bars = ax.bar(appraisal_metrics, appraisal_r2_values, color=['#2ca02c', '#d62728'], 
              width=0.6, edgecolor='black', linewidth=1.5)
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Method 1: Appraisal Dimension Prediction Performance', 
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target: 0.8')

# Add value labels
for bar, val in zip(bars, appraisal_r2_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('m1_appraisal_r2.png', dpi=300, bbox_inches='tight')
print("✓ Saved: m1_appraisal_r2.png")
plt.close()

# ==============================================================================
# FIGURE 3: 6D Appraisal Profiles - Radar Chart
# ==============================================================================
fig3 = plt.figure(figsize=(16, 10))

emotions = model_result_df['Emotion'].tolist()
dimensions = ['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power', 
              'Intrinsic_unpredictability', 'Normative_significance']

# Create subplots for different emotions
n_emotions = len(emotions)
n_cols = 4
n_rows = (n_emotions + n_cols - 1) // n_cols

for idx, emotion in enumerate(emotions, 1):
    ax = plt.subplot(n_rows, n_cols, idx, projection='polar')
    
    values = model_result_df[model_result_df['Emotion'] == emotion][dimensions].values[0].tolist()
    values += values[:1]  # Complete the circle
    
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, size=8)
    ax.set_ylim(0, 1)
    ax.set_title(emotion, fontsize=11, fontweight='bold', pad=20)
    ax.grid(True)

plt.suptitle('6D Appraisal Profiles - All Emotions (Method 1)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('m1_6d_appraisal_profiles.png', dpi=300, bbox_inches='tight')
print("✓ Saved: m1_6d_appraisal_profiles.png")
plt.close()

# ==============================================================================
# FIGURE 4: 6D Appraisal Heatmap
# ==============================================================================
fig4, ax = plt.subplots(figsize=(10, 8))

heatmap_data = model_result_df.set_index('Emotion')[dimensions]

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
            cbar_kws={'label': 'Appraisal Value'}, ax=ax, 
            linewidths=0.5, linecolor='gray')
ax.set_title('6D Appraisal Dimensions Heatmap - All Emotions', 
             fontsize=13, fontweight='bold')
ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
ax.set_xlabel('Appraisal Dimensions', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('m1_6d_appraisal_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: m1_6d_appraisal_heatmap.png")
plt.close()

# ==============================================================================
# FIGURE 5: Detailed Predictions by Experiment
# ==============================================================================
fig5, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, exp in enumerate(['Exp1_2', 'Exp3']):
    exp_data = emotion_pred_df[emotion_pred_df['Experiment'] == exp]
    
    emotions_exp = exp_data['Emotion'].tolist()
    predicted = exp_data['Predicted'].tolist()
    confidence = exp_data['Confidence'].tolist()
    correct = exp_data['Correct'].tolist()
    
    colors_bar = ['#2ca02c' if c else '#d62728' for c in correct]
    
    x_pos = np.arange(len(emotions_exp))
    bars = axes[ax_idx].bar(x_pos, confidence, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    axes[ax_idx].set_ylabel('Confidence Score', fontsize=11, fontweight='bold')
    axes[ax_idx].set_xlabel('Emotion', fontsize=11, fontweight='bold')
    axes[ax_idx].set_title(f'{exp} - Prediction Confidence', fontsize=12, fontweight='bold')
    axes[ax_idx].set_xticks(x_pos)
    axes[ax_idx].set_xticklabels(emotions_exp, rotation=45, ha='right')
    axes[ax_idx].set_ylim(0, 1.05)
    axes[ax_idx].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, conf, pred, correct_val in zip(bars, confidence, predicted, correct):
        height = bar.get_height()
        status = '✓' if correct_val else '✗'
        axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height,
                         f'{conf:.3f}\n{status}',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('m1_predictions_confidence.png', dpi=300, bbox_inches='tight')
print("✓ Saved: m1_predictions_confidence.png")
plt.close()

# ==============================================================================
# FIGURE 6: Combined Dashboard
# ==============================================================================
fig6 = plt.figure(figsize=(16, 10))
gs = fig6.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Accuracy comparison
ax1 = fig6.add_subplot(gs[0, :])
experiments_all = ['Exp1_2', 'Exp3']
accuracies_all = [exp1_2_acc, exp3_acc]
bars = ax1.bar(experiments_all, accuracies_all, color=['#1f77b4', '#ff7f0e'], 
               width=0.5, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Method 1: DQN + 6D Appraisal - Complete Results Dashboard', 
              fontsize=13, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accuracies_all):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

# 2. Appraisal R²
ax2 = fig6.add_subplot(gs[1, 0])
r2_vals = [appraisal_r2_values[0], appraisal_r2_values[1]]
bars = ax2.bar(['Exp1_2', 'Exp3'], r2_vals, color=['#2ca02c', '#d62728'], 
               edgecolor='black', linewidth=1.5)
ax2.set_ylabel('R² Score', fontsize=10, fontweight='bold')
ax2.set_title('Appraisal R² Scores', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, r2_vals):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Summary statistics
ax3 = fig6.add_subplot(gs[1, 1])
ax3.axis('off')
summary_text = f"""
PERFORMANCE SUMMARY

Exp1_2:
  • Accuracy: {exp1_2_acc:.2f}%
  • Correct: {exp1_2_correct}/{exp1_2_total}
  • Appraisal R²: {appraisal_r2_values[0]:.4f}

Exp3:
  • Accuracy: {exp3_acc:.2f}%
  • Correct: {exp3_correct}/{exp3_total}
  • Appraisal R²: {appraisal_r2_values[1]:.4f}

EMOTIONS: {len(emotions)}
DIMENSIONS: {len(dimensions)}
"""
ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, 
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Dimension importance
ax4 = fig6.add_subplot(gs[2, :])
dim_means = heatmap_data.mean().sort_values(ascending=False)
bars = ax4.barh(dim_means.index, dim_means.values, color='#9467bd', 
                 edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Mean Appraisal Value', fontsize=10, fontweight='bold')
ax4.set_title('Average Appraisal Dimension Activation', fontsize=11, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, dim_means.values)):
    ax4.text(val, i, f' {val:.3f}', va='center', fontweight='bold', fontsize=9)

plt.savefig('m1_complete_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: m1_complete_dashboard.png")
plt.close()

print("\n" + "="*60)
print("All graphs generated successfully!")
print("="*60)
print("Generated files:")
print("  1. m1_accuracy_comparison.png")
print("  2. m1_appraisal_r2.png")
print("  3. m1_6d_appraisal_profiles.png (radar charts)")
print("  4. m1_6d_appraisal_heatmap.png")
print("  5. m1_predictions_confidence.png")
print("  6. m1_complete_dashboard.png")
