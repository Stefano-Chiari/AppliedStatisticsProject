import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Lasso, Ridge


class LinearRegression:
    def __init__(self, penalty=None, alpha=1.0, fit_intercept=True):
        """
        Gestisce diversi modelli lineari.
        - penalty=None : Standard Ordinary Least Squares (OLS)
        - penalty='l1' : Lasso Regression (Feature selection implicita)
        - penalty='l2' : Ridge Regression
        """
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.is_fitted = False
        
        # Selezione del modello matematico
        if self.penalty == 'l1':
            self.model = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept, random_state=42)
        elif self.penalty == 'l2':
            self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, random_state=42)
        elif self.penalty is None:
            self.model = SklearnLR(fit_intercept=self.fit_intercept)
        else:
            raise ValueError("Il parametro 'penalty' deve essere None, 'l1' o 'l2'")

    def fit(self, X, y):
        """
        Addestra il modello calcolando i coefficienti.
        """
        # Conversione sicura in array NumPy
        X_arr = np.array(X)
        y_arr = np.array(y)
        
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

    def predict(self, X):
        """
        Effettua previsioni. 
        Gestisce in automatico sia matrici intere che singoli array 1D.
        """
        if not self.is_fitted:
            raise Exception("Errore critico: il modello non è stato addestrato. Eseguire .fit(X, y) prima di .predict()")
            
        X_arr = np.array(X)
        
        # Scikit-learn richiede array 2D. Se l'input è un array 1D (singola riga), lo riformattiamo.
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            # Restituiamo un singolo valore scalare
            return self.model.predict(X_arr)[0] 
            
        # Previsione vettorializzata standard
        return self.model.predict(X_arr)
        
    @property
    def coefficients(self):
        """Proprietà per accedere facilmente ai pesi del modello per l'inferenza"""
        if self.is_fitted:
            return self.model.coef_
        return None