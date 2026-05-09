import csv

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


FEATURES_4D = ["Suddenness", "Goal_relevance", "Conduciveness", "Power"]


def load_dataset(data_file, features=FEATURES_4D):
    data = pd.read_csv(data_file)
    return data[features].values.astype(np.float32), data["Emotion"].values.tolist()


def load_model_results(model_result_file, features=FEATURES_4D):
    data = pd.read_csv(model_result_file)
    return data[features].values.astype(np.float32), data["Emotion"].values.tolist()


class MLPEnsembleClassifier:
    def __init__(self, input_dim, class_names, n_models=5, epochs=300, lr=5e-4):
        self.class_names = list(class_names)
        self.class_to_index = {label: index for index, label in enumerate(self.class_names)}
        self.n_models = n_models
        self.epochs = epochs
        self.lr = lr
        self.input_dim = input_dim
        self.models = []

    def fit(self, X, y):
        y_encoded = np.array([self.class_to_index[label] for label in y], dtype=np.int64)
        self.models = []
        for model_index in range(self.n_models):
            model = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=self.lr,
                    alpha=1e-4,
                    max_iter=self.epochs,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=20,
                    random_state=model_index * 17 + 7,
                ),
            )
            model.fit(X, y_encoded)
            self.models.append(model)

    def predict_proba(self, X):
        return np.mean([model.predict_proba(X) for model in self.models], axis=0)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return [self.class_names[index] for index in np.argmax(probabilities, axis=1)]


class SVMBaseline:
    def __init__(self, C):
        self.C = C
        self.model = make_pipeline(
            StandardScaler(),
            svm.SVC(kernel="linear", C=C, probability=True),
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return list(self.model[-1].classes_)


def write_svm_probability_table(output_file, c_samples, class_names, story_names, probabilities):
    with open(output_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["C", "Story", "Emotion", "Val"])
        for c_value, by_story in zip(c_samples, probabilities):
            for story_name, story_probs in zip(story_names, by_story):
                for class_name, probability in zip(class_names, story_probs):
                    writer.writerow([c_value, story_name, class_name, probability])


def write_mlp_probability_table(output_file, class_names, story_names, probabilities, predictions):
    with open(output_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Story", "Predicted", "Emotion", "Val"])
        for story_name, predicted, story_probs in zip(story_names, predictions, probabilities):
            for class_name, probability in zip(class_names, story_probs):
                writer.writerow([story_name, predicted, class_name, probability])
