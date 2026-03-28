import numpy as np
import pandas as pd

class Regression:
    """
    Handles continuous target prediction.
    Topics: Linear Models, Feature Selection, Bias-Variance Tradeoff.
    """
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target = target_column
        self.results = {}

    def feature_selection(self, method='lasso'):
        """Lasso (L1) or Stepwise selection."""
        pass

    def run_linear_model(self):
        """Ordinary Least Squares or Ridge."""
        pass

    def run_tree_regression(self):
        """Decision Tree Regressor or Ensembles."""
        pass

    def analyze_bias_variance(self):
        """Validation curves to show tradeoff."""
        pass

    def get_metrics(self):
        """MSE, MAE, R-squared."""
        pass