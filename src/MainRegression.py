import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importa i modelli che abbiamo creato
from Regression.LinearRegression import LinearRegression
from Regression.TreeRegression import TreeRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Regression:
    """
    Handles continuous target prediction.
    Topics: Linear Models, Feature Selection, Bias-Variance Tradeoff.
    """
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target = target_column
        self.results = {}
        
        # Separiamo internamente le feature (X) dal target (y)
        self.X = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]

    def feature_selection(self, method='lasso'):
        """Lasso (L1) selection: restituisce solo le feature rilevanti."""
        if method.lower() != 'lasso':
            raise NotImplementedError("Attualmente è supportata solo la selezione tramite Lasso.")
            
        # Addestriamo una Lasso fortemente penalizzata (alpha alto) per scremare le variabili
        model = LinearRegression(penalty='l1', alpha=0.1)
        model.fit(self.X, self.y)
        
        # Estraiamo i coefficienti e teniamo solo quelli sopravvissuti (!= 0)
        coef = pd.Series(model.coefficients, index=self.X.columns)
        selected_features = coef[coef != 0].index.tolist()
        
        print(f"Feature Selection completata: mantenute {len(selected_features)} variabili su {len(self.X.columns)}.")
        return selected_features

    def run_linear_model(self, X_train, y_train, X_test, penalty=None, alpha=1.0):

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)  # transform, NON fit_transform!

        model = LinearRegression(penalty=penalty, alpha=alpha)
        model.fit(X_train_s, y_train)
        return model.predict(X_test_s)

    def run_tree_regression(self, X_train, y_train, X_test, n_estimators=100, max_depth=20):
        """Random Forest Regressor."""
        model = TreeRegression(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        return predictions

    def analyze_bias_variance(self):
        """
        Validation curves to show tradeoff.
        Genera un grafico che mostra come varia l'errore al variare della complessità del modello.
        """
        from sklearn.linear_model import Ridge
        
        print("Calcolo della Validation Curve in corso...")
        # Testiamo valori di penalità (alpha) da molto bassi (overfitting) a molto alti (underfitting)
        param_range = np.logspace(-3, 3, 7)
        
        # validation_curve esegue in automatico la cross-validation
        train_scores, test_scores = validation_curve(
            Ridge(), self.X, self.y, 
            param_name="alpha", param_range=param_range,
            cv=5, scoring="neg_mean_squared_error", n_jobs=-1
        )
        
        # Calcoliamo le medie degli errori (moltiplichiamo per -1 perché scikit-learn usa punteggi negativi per le loss)
        train_error = -np.mean(train_scores, axis=1)
        test_error = -np.mean(test_scores, axis=1)
        
        # Generiamo il grafico del Tradeoff Bias-Varianza
        plt.figure(figsize=(8, 6))
        plt.plot(param_range, train_error, label="Training Error (Bias)", color="blue", marker='o')
        plt.plot(param_range, test_error, label="Cross-Validation Error (Varianza)", color="red", marker='s')
        
        plt.xscale("log")
        plt.xlabel("Parametro Alpha (Regolarizzazione Ridge)")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("Bias-Variance Tradeoff")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def get_metrics(self, y_true, y_pred):
        """Ritorna un dizionario con MSE, MAE e R-squared."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MSE': round(mse, 4),
            'MAE': round(mae, 4),
            'R-squared': round(r2, 4)
        }
        return metrics
    


    def evaluate(self, model_type='linear', test_size=0.2, penalty=None, alpha=1.0,
                 n_estimators=100, max_depth=20, use_feature_selection=False):
        """
        Orchestrates: split → (optional) feature selection → train → predict → metrics.
        """
        X = self.X
        y = self.y

        if use_feature_selection:
            selected = self.feature_selection()
            X = X[selected]

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
        )

        if model_type == 'linear':
             y_pred = self.run_linear_model(X_train, y_train, X_test, penalty=penalty, alpha=alpha)
        elif model_type == 'tree':
            y_pred = self.run_tree_regression(X_train, y_train, X_test,
                                          n_estimators=n_estimators, max_depth=max_depth)
        else:
            raise ValueError("model_type deve essere 'linear' o 'tree'")

        metrics = self.get_metrics(y_test, y_pred)
        self.results[model_type] = metrics
        return metrics