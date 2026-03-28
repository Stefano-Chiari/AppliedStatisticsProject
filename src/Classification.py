import numpy as np
import pandas as pd

class Classification:
    """
    Handles categorical target prediction.
    Topics: Logistic Regression, Trees, Ensembles, Neural Networks.
    """
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target = target_column
        self.models = {} # Store different fitted classifiers

    def scale_and_split(self):
        """Train/Test split specifically for classification."""
        pass

    def run_logistic_regression(self):
        """Baseline linear classification."""
        pass

    def run_ensemble_methods(self):
        """Random Forest or Gradient Boosting."""
        pass

    def run_neural_network(self):
        """Multi-layer Perceptron (MLP)."""
        pass

    def evaluate_performance(self):
        """ROC Curves, Confusion Matrix, Accuracy/F1."""
        pass