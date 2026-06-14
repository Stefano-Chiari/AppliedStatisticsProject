import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importa i modelli che abbiamo creato
from Regression.LinearRegression import LinearRegression
from Regression.TreeRegression import TreeRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

class Regression:
    """
    Handles continuous target prediction.
    Topics: Linear Models, Feature Selection, Bias-Variance Tradeoff.
    """
    def __init__(self, df: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
        self.df = df
        self.target = target_column
        self.results = {}
        
        # 1. Separiamo feature (X) e target (y) dal dataframe originale
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # 2. Dividiamo i dati in set di training e test PRIMA di qualsiasi altra operazione
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Dati divisi in training ({len(self.X_train)} righe) e test ({len(self.X_test)} righe).")

    def _feature_selection(self, X_train, y_train, method='lasso', alpha=0.001):
        """Lasso (L1) selection: addestrato SOLO sul training set per scegliere le feature."""
        if method.lower() != 'lasso':
            raise NotImplementedError("Attualmente è supportata solo la selezione tramite Lasso.")
        
        # Per la selezione delle feature, è cruciale scalare i dati per rendere i coefficienti confrontabili
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Estraiamo i coefficienti e teniamo solo quelli sopravvissuti (!= 0)
        coef = pd.Series(model.coef_, index=X_train.columns)
        selected_features = coef[coef != 0].index.tolist()
        
        print(f"Feature Selection completata: mantenute {len(selected_features)} variabili su {len(X_train.columns)}.")
        return selected_features

    def run_linear_model(self, X_train, y_train, X_test, penalty=None, alpha=1.0):

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

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
        Usa una versione semplificata dei dati di training (solo numerici) per l'analisi.
        """
        print("Calcolo della Validation Curve in corso...")

        # Usiamo solo le feature numeriche del training set per questa analisi
        X_train_numeric = self.X_train.select_dtypes(include=np.number).fillna(0)
        y_train_log = np.log1p(self.y_train) # Il target va trasformato in log per coerenza

        # I dati per Ridge devono essere scalati
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_numeric)

        # Testiamo valori di penalità (alpha) da molto bassi (overfitting) a molto alti (underfitting)
        param_range = np.logspace(-3, 3, 7)
        
        # validation_curve esegue in automatico la cross-validation
        train_scores, test_scores = validation_curve(
            Ridge(), X_train_s, y_train_log, 
            param_name="alpha", param_range=param_range,
            cv=5, scoring="neg_mean_squared_error", n_jobs=-1
        )
        
        # Calcoliamo le medie degli errori (moltiplichiamo per -1 perché scikit-learn usa punteggi negativi per le loss)
        train_error = -np.mean(train_scores, axis=1)
        test_error  = -np.mean(test_scores,  axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(param_range, train_error, label="Training Error (Bias)",
                color="blue", marker='o')
        plt.plot(param_range, test_error,  label="Cross-Validation Error (Varianza)",
                color="red",  marker='s')

        plt.xscale("log")
        plt.xlabel("Parametro Alpha (Regolarizzazione Ridge)")
        plt.ylabel("Mean Squared Error (su scala logaritmica)")
        plt.title("Bias-Variance Tradeoff")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.6)

        # Ylim calcolato sui dati reali, non su un margin fisso
        all_errors = np.concatenate([train_error, test_error])
        plt.ylim(all_errors.min() * 0.999, all_errors.max() * 1.001)

        plt.show()

    def get_metrics(self, y_true, y_pred):
        """Ritorna un dizionario con MSE, MAE e R-squared."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'RMSE': round(np.sqrt(mse), 4), # Spesso si usa RMSE per avere la stessa unità di misura del target
            'MAE': round(mae, 4),
            'R-squared': round(r2, 4),
        }
        return metrics
    
    def evaluate(self, model_type='linear', penalty=None, alpha=1.0,
             n_estimators=100, max_depth=20, use_feature_selection=False):
        """
        Orchestra: (opzionale) feature selection → training → predizione → metriche.
        Il preprocessing va fatto esternamente prima di istanziare Regression.
        """
        X_train = self.X_train
        X_test  = self.X_test
        y_train = self.y_train
        y_test  = self.y_test

        if use_feature_selection:
            selected_features = self._feature_selection(X_train, y_train)
            X_train = X_train[selected_features]
            X_test  = X_test[selected_features]

        if model_type == 'linear':
            y_pred = self.run_linear_model(X_train, y_train, X_test,
                                        penalty=penalty, alpha=alpha)
        elif model_type == 'tree':
            y_pred = self.run_tree_regression(X_train, y_train, X_test,
                                            n_estimators=n_estimators,
                                            max_depth=max_depth)
        else:
            raise ValueError("model_type deve essere 'linear' o 'tree'")

        # Il target è log_price, quindi riconvertiamo per le metriche in dollari
        y_pred_dollars = np.expm1(y_pred)
        y_test_dollars = np.expm1(y_test)

        metrics = self.get_metrics(y_test_dollars, y_pred_dollars)
        self.results[model_type] = metrics
        print(f"Risultati '{model_type}': {metrics}")
        return metrics

    def plot_feature_importance(self, top_n=15):
        """
        Mostra i top driver per Random Forest e Lasso.
        Usa i dati già splittati nell'__init__.
        """
        X_train = self.X_train
        y_train = self.y_train

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Primary Drivers of Price", fontsize=14, fontweight='bold')

        # Random Forest
        tree_model = TreeRegression(n_estimators=100, max_depth=20)
        tree_model.fit(X_train, y_train)
        importances = tree_model.feature_importances
        importances.nlargest(top_n).sort_values().plot(
            kind='barh', ax=axes[0], color='steelblue'
        )
        axes[0].set_title("Random Forest — Feature Importances")
        axes[0].set_xlabel("Importance")
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Lasso
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        lasso_model = LinearRegression(penalty='l1', alpha=0.001)
        lasso_model.fit(X_train_scaled, y_train)
        coefs = pd.Series(lasso_model.coefficients, index=X_train.columns)
        coefs_nonzero = coefs[coefs != 0]
        top_coefs = coefs_nonzero.reindex(
            coefs_nonzero.abs().nlargest(top_n).index
        ).sort_values()
        colors = ['tomato' if x < 0 else 'steelblue' for x in top_coefs]
        top_coefs.plot(kind='barh', ax=axes[1], color=colors)
        axes[1].set_title("Lasso — Coefficienti")
        axes[1].set_xlabel("Coefficiente (su log_price)")
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
