"""
neural_classifier.py - Neural Network Emotion Classifier
==========================================================
IMPROVEMENT #3: Replaces the SVM classifier with an ensemble of
MLP classifiers for more robust emotion prediction.

Key improvements over original SVM:
  - 3-layer MLP (input→64→32→n_classes) with ReLU and dropout
  - Ensemble of 10 models for variance reduction
  - 5-fold cross-validation for hyperparameter-free evaluation
  - Supports both 4D (original) and 6D (extended) appraisal vectors
  - Retains SVM baseline for direct comparison
"""

import numpy as np
import csv
from scipy.stats import halfnorm
from sklearn import svm
from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Data Generation (Scherer's CPM patterns)
# =============================================================================

GENERATORS = {
    'very_low': lambda: round(float(halfnorm.rvs(loc=0, scale=0.05)), 5),
    'obstruct': lambda: round(float(halfnorm.rvs(loc=0, scale=0.05)), 5),
    'low':      lambda: round(float(halfnorm.rvs(loc=0, scale=0.1)), 5),
    'medium':   lambda: round(float(np.random.normal(loc=0.5, scale=0.05)), 5),
    'high':     lambda: round(float(1 - halfnorm.rvs(loc=0, scale=0.1)), 5),
    'very_high':lambda: round(float(1 - halfnorm.rvs(loc=0, scale=0.05)), 5),
    'open':     lambda: round(float(np.random.uniform(0, 1)), 5),
}

# Scherer's nominal patterns for 4 dimensions
# [Suddenness, Goal_relevance, Conduciveness, Power]
EXP12_PATTERNS_4D = {
    'Boredom':   ['very_low', 'low',    'open',     'medium'],
    'Fear':      ['high',     'high',   'obstruct', 'very_low'],
    'Happiness': ['low',      'medium', 'high',     'open'],
    'Joy':       ['high',     'high',   'very_high','open'],
    'Pride':     ['open',     'high',   'high',     'open'],
    'Sadness':   ['low',      'high',   'obstruct', 'very_low'],
    'Shame':     ['open',     'high',   'open',     'open'],
}

EXP3_PATTERNS_4D = {
    'Anxiety':    ['low',  'medium', 'obstruct', 'low'],
    'Despair':    ['high', 'high',   'obstruct', 'very_low'],
    'Irritation': ['low',  'medium', 'obstruct', 'medium'],
    'Rage':       ['high', 'high',   'obstruct', 'high'],
}

# Extended patterns for 6 dimensions:
# [Suddenness, Goal_relevance, Conduciveness, Power, Intrinsic_unpred, Normative_sig]
EXP12_PATTERNS_6D = {
    'Boredom':   ['very_low', 'low',    'open',     'medium',   'very_low', 'very_low'],
    'Fear':      ['high',     'high',   'obstruct', 'very_low', 'high',     'high'],
    'Happiness': ['low',      'medium', 'high',     'open',     'low',      'medium'],
    'Joy':       ['high',     'high',   'very_high','open',     'high',     'very_high'],
    'Pride':     ['open',     'high',   'high',     'open',     'medium',   'high'],
    'Sadness':   ['low',      'high',   'obstruct', 'very_low', 'low',      'medium'],
    'Shame':     ['open',     'high',   'open',     'open',     'medium',   'medium'],
}

EXP3_PATTERNS_6D = {
    'Anxiety':    ['low',  'medium', 'obstruct', 'low',      'medium',   'medium'],
    'Despair':    ['high', 'high',   'obstruct', 'very_low', 'high',     'high'],
    'Irritation': ['low',  'medium', 'obstruct', 'medium',   'medium',   'low'],
    'Rage':       ['high', 'high',   'obstruct', 'high',     'high',     'very_high'],
}


import pandas as pd

def generate_training_data(patterns, n_per_class=400, csv_path=None):
    """Generate synthetic training data from Scherer patterns or load from CSV.

    If csv_path is provided, loads the first 4 dimensions from the CSV and
    generates the remaining dimensions (if any) using the patterns.

    Returns:
        X: np.array of shape (n_total, n_dims)
        y: list of emotion labels
        feature_names: list of feature names
    """
    sample_entry = list(patterns.values())[0]
    n_dims = len(sample_entry)

    if n_dims == 4:
        feature_names = ['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power']
    else:
        feature_names = ['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power',
                         'Intrinsic_unpredictability', 'Normative_significance']

    if csv_path:
        df = pd.read_csv(csv_path)
        # Filter dataframe to only include emotions present in patterns
        df = df[df['Emotion'].isin(patterns.keys())]
        y_labels = df['Emotion'].tolist()
        X_4d = df[['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power']].values.astype(np.float32)
        
        if n_dims == 4:
            return X_4d, y_labels, feature_names
            
        X_rows = []
        for i, row_4d in enumerate(X_4d):
            emo = y_labels[i]
            row = list(row_4d)
            for d in range(4, n_dims):
                val = GENERATORS[patterns[emo][d]]()
                row.append(max(0.0, min(1.0, val)))
            X_rows.append(row)
        return np.array(X_rows, dtype=np.float32), y_labels, feature_names

    X_rows = []
    y_labels = []

    for emotion, dims in patterns.items():
        for _ in range(n_per_class):
            row = [GENERATORS[d]() for d in dims]
            # Clip to [0, 1]
            row = [max(0.0, min(1.0, v)) for v in row]
            X_rows.append(row)
            y_labels.append(emotion)

    return np.array(X_rows, dtype=np.float32), y_labels, feature_names


# =============================================================================
# Neural Classifier
# =============================================================================

class EmotionMLP(nn.Module):
    """3-layer MLP for emotion classification."""

    def __init__(self, input_dim, n_classes, hidden1=64, hidden2=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class NeuralClassifierEnsemble:
    """Ensemble of MLP classifiers for robust emotion prediction.

    Trains n_models MLPs with different random seeds and averages
    softmax probabilities for the final prediction.
    """

    def __init__(self, input_dim, class_names, n_models=10, epochs=200, lr=1e-3):
        self.class_names = list(class_names)
        self.n_classes = len(self.class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.input_dim = input_dim
        self.n_models = n_models
        self.epochs = epochs
        self.lr = lr
        self.models = []

    def fit(self, X, y):
        """Train ensemble of MLPs.

        Args:
            X: np.array (n_samples, input_dim)
            y: list of emotion labels
        """
        y_idx = np.array([self.class_to_idx[label] for label in y])
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y_idx)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        self.models = []
        for i in range(self.n_models):
            torch.manual_seed(i * 42)
            model = EmotionMLP(self.input_dim, self.n_classes)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for epoch in range(self.epochs):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    out = model(X_batch)
                    loss = criterion(out, y_batch)
                    loss.backward()
                    optimizer.step()

            model.eval()
            self.models.append(model)

    def predict_proba(self, X):
        """Predict class probabilities by averaging over the ensemble.

        Args:
            X: np.array (n_samples, input_dim)

        Returns:
            np.array (n_samples, n_classes) of averaged probabilities
        """
        X_tensor = torch.FloatTensor(X)
        all_probs = []

        for model in self.models:
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.softmax(logits, dim=1).numpy()
                all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return [self.class_names[i] for i in np.argmax(probs, axis=1)]


# =============================================================================
# SVM Baseline (same as original paper)
# =============================================================================

class SVMBaseline:
    """SVM classifier matching the original paper implementation."""

    def __init__(self, C=0.0032, class_names=None):
        self.C = C
        self.class_names = class_names
        self.model = None

    def fit(self, X, y):
        self.model = svm.SVC(kernel='linear', C=self.C, probability=True)
        self.model.fit(X, y)
        if self.class_names is None:
            self.class_names = list(self.model.classes_)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def cross_val_accuracy(self, X, y, cv=5):
        """Compute cross-validated accuracy."""
        scores = cross_val_score(
            svm.SVC(kernel='linear', C=self.C),
            X, y, cv=cv, scoring='accuracy'
        )
        return scores.mean(), scores.std()
