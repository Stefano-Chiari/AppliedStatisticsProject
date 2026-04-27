import numpy as np
from sklearn.ensemble import RandomForestRegressor

class TreeRegression:
    def __init__(self, n_estimators=100, max_depth=None):
        """
        Gestisce modelli ensemble basati su alberi decisionali.
        - n_estimators: numero di alberi nella foresta.
        - max_depth: profondità massima (utile per evitare overfitting).
        """
        # n_jobs=-1 usa tutti i core del processore per parallelizzare l'addestramento
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42, 
            n_jobs=-1 
        )
        self.is_fitted = False

    def fit(self, X, y):
        """Addestra la foresta casuale."""
        X_arr = np.array(X)
        y_arr = np.array(y)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

    def predict(self, X):
        """Effettua le previsioni."""
        if not self.is_fitted:
            raise Exception("Errore: il modello TreeRegression non è stato addestrato.")
        
        X_arr = np.array(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            return self.model.predict(X_arr)[0]
            
        return self.model.predict(X_arr)

    @property
    def feature_importances(self):
        """
        Negli alberi non ci sono coefficienti (pesi), ma un indice di 
        'importanza' che indica quanto ogni feature riduca l'errore.
        """
        if self.is_fitted:
            return self.model.feature_importances_
        return None