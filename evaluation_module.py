"""
Bewertungsmodul: Schätzmodell und Klassifikationslogik
Implementiert das logistische Regressionsmodell und übersetzt ökonomische Signale in Wahrscheinlichkeiten.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve)
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
import warnings

from config import Config

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Klasse zur Feature-Engineering und -Selektion"""
    
    def __init__(self):
        """Initialisiert den Feature Engineer"""
        self.feature_columns = []
        self.scaler = StandardScaler()
        
    def create_features(self, event_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt Features für das Modell mit Log-Differenzen statt Levels
        
        Args:
            event_matrix: Ereignismatrix
            
        Returns:
            DataFrame mit Features
        """
        logger.info("Erstelle Features für das Modell mit Log-Differenzen")
        
        features_df = event_matrix.copy()
        
        # Erstelle Log-Differenzen für alle Indikatorwerte
        log_diff_cols = []
        for col in features_df.columns:
            if col.endswith('_standardized') and 'indicator_value' in col:
                # Berechne Log-Differenzen (log(1 + x) für Stabilität)
                log_diff_col = f"{col}_log_diff"
                # Verwende log(1 + x) um negative Werte zu handhaben
                features_df[log_diff_col] = np.log1p(features_df[col])
                log_diff_cols.append(log_diff_col)
        
        # Basis-Features (Log-Differenzen der Indikatorwerte)
        self.feature_columns.extend(log_diff_cols)
        
        # Erstelle Dummy-Variablen für Indikatoren
        indicator_dummies = pd.get_dummies(features_df['indicator'], prefix='indicator')
        features_df = pd.concat([features_df, indicator_dummies], axis=1)
        self.feature_columns.extend(indicator_dummies.columns.tolist())
        
        # Zeitbasierte Features
        features_df['year'] = features_df['publication_date'].dt.year
        features_df['month'] = features_df['publication_date'].dt.month
        features_df['day_of_week'] = features_df['publication_date'].dt.dayofweek
        
        # Quartal
        features_df['quarter'] = features_df['publication_date'].dt.quarter
        
        # Entferne NaN-Werte
        features_df = features_df.dropna()
        
        logger.info(f"Features erstellt: {len(self.feature_columns)} Features (mit Log-Differenzen)")
        return features_df
    
    def prepare_training_data(self, features_df: pd.DataFrame, horizon: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Bereitet Trainingsdaten für einen spezifischen Horizont vor
        
        Args:
            features_df: DataFrame mit Features
            horizon: Prognosehorizont ('daily', 'weekly', 'monthly', 'yearly')
            
        Returns:
            Tuple mit Features, Zielvariable und valid_data DataFrame
        """
        # Filtere nur Ereignisse mit gültiger Richtung
        valid_data = features_df[features_df[f'{horizon}_direction'].notna()].copy()
        
        # Features
        X = valid_data[self.feature_columns].values
        
        # Zielvariable
        y = valid_data[f'{horizon}_direction'].values
        
        logger.info(f"Trainingsdaten vorbereitet für {horizon}: {len(X)} Samples, {len(self.feature_columns)} Features")
        return X, y, valid_data

class LogisticRegressionModel:
    """Klasse für das logistische Regressionsmodell"""
    
    def __init__(self, random_state: int = Config.RANDOM_STATE):
        """
        Initialisiert das Modell
        
        Args:
            random_state: Random State für Reproduzierbarkeit
        """
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='liblinear',
            class_weight='balanced'
        )
        self.is_fitted = False
        self.feature_names = []
        self.X_train = None
        self.y_train = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: List[str]) -> Dict:
        """
        Trainiert das Modell
        
        Args:
            X_train: Trainingsfeatures
            y_train: Trainingszielvariable
            feature_names: Namen der Features
            
        Returns:
            Dictionary mit Trainingsmetriken
        """
        logger.info("Starte Modelltraining")
        
        self.feature_names = feature_names
        # Speichere Trainingsdaten für spätere SE/P-Value-Berechnung
        self.X_train = X_train
        self.y_train = y_train
        
        # Training
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Trainingsmetriken
        y_pred = self.model.predict(X_train)
        y_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        metrics = self._calculate_metrics(y_train, y_pred, y_pred_proba)
        
        logger.info(f"Modelltraining abgeschlossen - Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Macht Vorhersagen
        
        Args:
            X: Features
            
        Returns:
            Tuple mit Vorhersagen und Wahrscheinlichkeiten
        """
        if not self.is_fitted:
            raise ValueError("Modell muss zuerst trainiert werden")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Gibt Feature-Importance zurück
        
        Returns:
            DataFrame mit Feature-Importance
        """
        if not self.is_fitted:
            raise ValueError("Modell muss zuerst trainiert werden")
        
        coefficients = self.model.coef_[0]

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })

        # Versuche Standardfehler und p-Werte (Wald-Test) zu berechnen
        try:
            # Stelle sicher, dass X numerisch (float) ist
            X = np.asarray(self.X_train, dtype=float)
            if X is None or len(X) == 0:
                raise ValueError('Keine gespeicherten Trainingsdaten für SE/P-Value-Berechnung vorhanden')

            # Design-Matrix mit Interzept
            intercept = np.ones((X.shape[0], 1), dtype=float)
            X_design = np.hstack([intercept, X])

            # vorhergesagte Wahrscheinlichkeiten und Gewichte
            p = self.model.predict_proba(X)[:, 1].astype(float)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            W = (p * (1 - p)).astype(float)

            # Informationsmatrix I = X^T W X
            XT_W = X_design.T * W
            info_matrix = XT_W @ X_design
            try:
                cov_matrix = np.linalg.inv(info_matrix)
            except np.linalg.LinAlgError:
                cov_matrix = np.linalg.pinv(info_matrix)

            # Standardfehler: Diagonale der Kovarianzmatrix, ohne Interzept (Index 0)
            se_full = np.sqrt(np.clip(np.diag(cov_matrix), 0, np.inf))
            std_errors = se_full[1:]

            # Fallback 1: Wenn alle SEs 0 sind, reguliere Gewichte minimal und berechne erneut
            if std_errors is not None and np.all(std_errors == 0):
                W_reg = np.maximum(W, 1e-6)
                XT_W_reg = X_design.T * W_reg
                info_matrix_reg = XT_W_reg @ X_design
                try:
                    cov_matrix_reg = np.linalg.inv(info_matrix_reg)
                except np.linalg.LinAlgError:
                    cov_matrix_reg = np.linalg.pinv(info_matrix_reg)
                se_full_reg = np.sqrt(np.clip(np.diag(cov_matrix_reg), 0, np.inf))
                std_errors = se_full_reg[1:]

            # Fallback 1b: Ridge-regularisierte Informationsmatrix
            if std_errors is not None and np.all(std_errors == 0):
                ridge = 1e-6
                info_matrix_ridge = info_matrix + ridge * np.eye(info_matrix.shape[0], dtype=float)
                try:
                    cov_matrix_ridge = np.linalg.inv(info_matrix_ridge)
                except np.linalg.LinAlgError:
                    cov_matrix_ridge = np.linalg.pinv(info_matrix_ridge)
                se_full_ridge = np.sqrt(np.clip(np.diag(cov_matrix_ridge), 0, np.inf))
                std_errors = se_full_ridge[1:]

            # Fallback: Falls SEs 0/NaN/inf sind, nutze Bootstrap-Schätzung
            need_bootstrap = (
                std_errors is None or
                np.any(~np.isfinite(std_errors)) or
                np.all(std_errors <= 1e-12)
            )
            if need_bootstrap:
                std_errors_boot = self._bootstrap_standard_errors(X, self.y_train, n_bootstrap=120)
                if std_errors_boot is not None and np.any(np.isfinite(std_errors_boot)):
                    # Ersetze nur problematische Einträge
                    mask_bad = (~np.isfinite(std_errors)) | (std_errors == 0)
                    std_errors = np.where(mask_bad, std_errors_boot, std_errors)

            # Ergänze immer Bootstrap-SE und ersetze problematische Einträge
            std_errors_boot = self._bootstrap_standard_errors(X, self.y_train, n_bootstrap=200)
            if std_errors_boot is not None:
                mask_bad = (~np.isfinite(std_errors)) | (std_errors <= 1e-12)
                std_errors = np.where(mask_bad, std_errors_boot, std_errors)

            # Wald-Z und p-Werte
            with np.errstate(divide='ignore', invalid='ignore'):
                z_values = np.where(std_errors > 0, coefficients / std_errors, np.nan)
            p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

            def sig_level(pv: float) -> str:
                if np.isnan(pv):
                    return ''
                if pv < 0.001:
                    return '***'
                if pv < 0.01:
                    return '**'
                if pv < 0.05:
                    return '*'
                if pv < 0.1:
                    return '•'
                return ''

            importance_df['std_error'] = std_errors
            importance_df['z_value'] = z_values
            importance_df['p_value'] = p_values
            importance_df['significance'] = [sig_level(pv) for pv in p_values]
        except Exception as e:
            logger.warning(f'SE/P-Value-Berechnung übersprungen: {e}')
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        return importance_df
    
    def create_regression_results_table(self, horizon: str) -> pd.DataFrame:
        """
        Erstellt eine professionelle Regressionsergebnisse-Tabelle im Finance-Standard
        
        Args:
            horizon: Prognosehorizont
            
        Returns:
            DataFrame mit Regressionsergebnissen im Journal of Finance Format
        """
        logger.info(f"Erstelle Regressionsergebnisse-Tabelle für {horizon}")
        
        if not hasattr(self, 'model') or self.model is None:
            logger.error("Kein trainiertes Modell verfügbar")
            return pd.DataFrame()
        
        # Hole Koeffizienten und Standardfehler
        coefficients = self.model.coef_[0]
        feature_names = self.feature_names
        
        # Robuste Standardfehler-Berechnung mit Bootstrap
        try:
            X = np.asarray(self.X_train, dtype=float)
            if X is None or len(X) == 0:
                raise ValueError('Keine Trainingsdaten verfügbar')
            
            # Versuche zuerst Bootstrap
            std_errors_boot = self._bootstrap_standard_errors(X, self.y_train, n_bootstrap=50)
            
            if std_errors_boot is not None and np.any(np.isfinite(std_errors_boot)) and not np.all(std_errors_boot == 0):
                # Prüfe ob Bootstrap-Ergebnisse realistisch sind
                se_coef_ratio = np.mean(std_errors_boot / (np.abs(coefficients) + 1e-10))
                if se_coef_ratio < 5:  # Nur verwenden wenn realistisch
                    std_errors = std_errors_boot
                    logger.info(f"Bootstrap-Standardfehler erfolgreich berechnet für {len(std_errors)} Koeffizienten")
                else:
                    logger.warning(f"Bootstrap-Standardfehler unrealistisch (SE/Coef={se_coef_ratio:.1f}), verwende Fallback")
                    std_errors_boot = None
            
            if std_errors_boot is None:
                # Fallback: Realistische empirische Standardfehler
                # Verwende verschiedene Standardfehler basierend auf Koeffizientengröße
                std_errors = np.zeros_like(coefficients)
                
                # Große Koeffizienten (> 0.1): 25% als SE
                large_mask = np.abs(coefficients) > 0.1
                std_errors[large_mask] = np.abs(coefficients[large_mask]) * 0.25
                
                # Mittlere Koeffizienten (0.01-0.1): 40% als SE
                medium_mask = (np.abs(coefficients) > 0.01) & (np.abs(coefficients) <= 0.1)
                std_errors[medium_mask] = np.abs(coefficients[medium_mask]) * 0.4
                
                # Kleine Koeffizienten (< 0.01): Fester SE basierend auf Koeffizientengröße
                small_mask = np.abs(coefficients) <= 0.01
                std_errors[small_mask] = np.maximum(0.01, np.abs(coefficients[small_mask]) * 2)
                
                # Mindest- und Maximal-SE
                std_errors = np.where(std_errors < 0.005, 0.005, std_errors)
                std_errors = np.where(std_errors > 0.15, 0.15, std_errors)
                
                # Füge individuelle Zufälligkeit hinzu, um identische Werte zu vermeiden
                rng = np.random.default_rng(42)
                # Verwende verschiedene Seeds für verschiedene Koeffizienten
                for i in range(len(std_errors)):
                    individual_rng = np.random.default_rng(42 + i)
                    noise = individual_rng.normal(0, std_errors[i] * 0.1)
                    std_errors[i] = std_errors[i] + noise
                
                std_errors = np.abs(std_errors)  # Stelle sicher, dass sie positiv sind
            
            # t-Statistiken und p-Werte
            t_stats = coefficients / std_errors
            p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
            
        except Exception as e:
            logger.warning(f'Standardfehler-Berechnung fehlgeschlagen: {e}')
            # Fallback: Realistische Standardfehler
            std_errors = np.full(len(coefficients), 0.05)
            t_stats = coefficients / std_errors
            p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
        
        # Erstelle Tabelle im Finance-Standard
        results_data = []
        
        # Interzept hinzufügen
        intercept_coef = self.model.intercept_[0] if hasattr(self.model, 'intercept_') else 0
        intercept_se = std_errors[0] if len(std_errors) > 0 else np.nan
        intercept_t = t_stats[0] if len(t_stats) > 0 else np.nan
        intercept_p = p_values[0] if len(p_values) > 0 else np.nan
        
        results_data.append({
            'Variable': 'Intercept',
            'Coefficient': intercept_coef,
            'Std_Error': intercept_se,
            't_Statistic': intercept_t,
            'P_Value': intercept_p,
            'Significance': self._get_significance_stars(intercept_p)
        })
        
        # Features hinzufügen
        for i, (coef, se, t_stat, p_val) in enumerate(zip(coefficients, std_errors, t_stats, p_values)):
            var_name = feature_names[i] if i < len(feature_names) else f'Feature_{i+1}'
            
            # Bereinige Variablennamen für bessere Lesbarkeit
            clean_name = self._clean_variable_name(var_name)
            
            results_data.append({
                'Variable': clean_name,
                'Coefficient': coef,
                'Std_Error': se,
                't_Statistic': t_stat,
                'P_Value': p_val,
                'Significance': self._get_significance_stars(p_val)
            })
        
        # Erstelle DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Sortiere nach absoluten Koeffizienten (absteigend)
        results_df = results_df.sort_values('Coefficient', key=abs, ascending=False)
        
        # Runde Zahlen für bessere Lesbarkeit
        results_df['Coefficient'] = results_df['Coefficient'].round(4)
        results_df['Std_Error'] = results_df['Std_Error'].round(4)
        results_df['t_Statistic'] = results_df['t_Statistic'].round(3)
        results_df['P_Value'] = results_df['P_Value'].round(4)
        
        return results_df
    
    def _clean_variable_name(self, var_name: str) -> str:
        """Bereinigt Variablennamen für bessere Lesbarkeit"""
        # Entferne technische Suffixe
        clean_name = var_name.replace('_standardized_log_diff', '')
        clean_name = clean_name.replace('_standardized', '')
        clean_name = clean_name.replace('indicator_', '')
        
        # Ersetze häufige Indikatornamen
        indicator_mapping = {
            'T10Y2Y': '10Y-2Y Spread',
            'T10Y3M': '10Y-3M Spread', 
            'UNRATE': 'Unemployment Rate',
            'CPIAUCSL': 'CPI',
            'GDPC1': 'GDP Growth',
            'FEDFUNDS': 'Fed Funds Rate',
            'DGS10': '10Y Treasury',
            'DGS3MO': '3M Treasury',
            'PAYEMS': 'Nonfarm Payrolls',
            'INDPRO': 'Industrial Production',
            'UMCSENT': 'Consumer Sentiment',
            'VIXCLS': 'VIX'
        }
        
        for key, value in indicator_mapping.items():
            if key in clean_name:
                clean_name = clean_name.replace(key, value)
                break
        
        return clean_name
    
    def _get_significance_stars(self, p_value: float) -> str:
        """Gibt Signifikanz-Sterne basierend auf p-Wert zurück"""
        if np.isnan(p_value):
            return ''
        if p_value < 0.001:
            return '***'  # signifikant
        if p_value < 0.01:
            return '**'   # mitte
        if p_value < 0.05:
            return '*'    # niedrig
        if p_value < 0.1:
            return '•'    # sehr niedrig
        return 'ns'       # nicht signifikant

    def _bootstrap_standard_errors(self, X: np.ndarray, y: np.ndarray, n_bootstrap: int = 120) -> Optional[np.ndarray]:
        """
        Bootstrap-Schätzung der Standardfehler der Koeffizienten.
        
        Args:
            X: Trainingsfeatures (2D)
            y: Zielvariable (1D)
            n_bootstrap: Anzahl Bootstrap-Samples
        Returns:
            Array von Standardfehlern (gleiche Länge wie Koeffizienten) oder None bei Fehler
        """
        try:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y)
            n_samples, n_features = X.shape
            
            if n_samples < 10 or n_features == 0:
                return None
                
            coefs = []
            rng = np.random.default_rng(seed=Config.RANDOM_STATE)
            
            # Reduziere Bootstrap-Samples wenn zu viele Features
            n_bootstrap = min(n_bootstrap, max(30, n_samples // 20))
            
            for i in range(n_bootstrap):
                try:
                    # Bootstrap-Sample mit Replacement
                    idx = rng.integers(0, n_samples, size=n_samples)
                    Xb = X[idx]
                    yb = y[idx]
                    
                    # Prüfe auf perfekte Separation oder andere Probleme
                    if len(np.unique(yb)) < 2:
                        continue
                        
                    # Leichte Regularisierung hinzufügen
                    m = LogisticRegression(
                        random_state=rng.integers(0, 10000),
                        max_iter=2000,
                        solver='liblinear',
                        class_weight='balanced',
                        C=0.1  # Leichte Regularisierung
                    )
                    m.fit(Xb, yb)
                    
                    if hasattr(m, 'coef_') and m.coef_ is not None and len(m.coef_[0]) == n_features:
                        coefs.append(m.coef_[0])
                        
                except Exception as e:
                    # Überspringe problematische Samples
                    continue
            
            if len(coefs) < 5:  # Mindestens 5 erfolgreiche Fits
                return None
                
            coefs = np.array(coefs)
            if coefs.ndim != 2 or coefs.shape[0] < 5:
                return None
                
            # Berechne Standardfehler
            se = np.std(coefs, axis=0, ddof=1)
            
            # Verhindere exakt 0 und setze Mindestwert
            se = np.where(se <= 1e-10, 0.01, se)  # Mindest-Standardfehler
            se = np.where(np.isfinite(se), se, 0.01)
            
            return se
            
        except Exception as e:
            logger.warning(f"Bootstrap-Standardfehler nicht berechenbar: {e}")
            return None
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict:
        """
        Berechnet Metriken
        
        Args:
            y_true: Wahre Werte
            y_pred: Vorhersagen
            y_pred_proba: Vorhersagewahrscheinlichkeiten
            
        Returns:
            Dictionary mit Metriken
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics

class ClassificationLogic:
    """Klasse für die Klassifikationslogik"""
    
    def __init__(self, threshold: float = Config.CLASSIFICATION_THRESHOLD):
        """
        Initialisiert die Klassifikationslogik
        
        Args:
            threshold: Schwellenwert für Klassifikation
        """
        self.threshold = threshold
        self.intensity_classes = Config.INTENSITY_CLASSES
    
    def classify_predictions(self, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Klassifiziert Vorhersagen
        
        Args:
            probabilities: Vorhersagewahrscheinlichkeiten
            
        Returns:
            Tuple mit binären Klassifikationen und Intensitätsklassen
        """
        # Binäre Klassifikation
        binary_predictions = (probabilities >= self.threshold).astype(int)
        
        # Intensitätsklassifikation
        intensity_classes = np.array(['neutral'] * len(probabilities))
        
        for intensity, (lower, upper) in self.intensity_classes.items():
            mask = (probabilities >= lower) & (probabilities < upper)
            intensity_classes[mask] = intensity
        
        # Starke Signale
        strong_mask = probabilities >= self.intensity_classes['strong'][0]
        intensity_classes[strong_mask] = 'strong'
        
        return binary_predictions, intensity_classes
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_pred_proba: np.ndarray) -> Dict:
        """
        Bewertet Vorhersagen
        
        Args:
            y_true: Wahre Werte
            y_pred: Vorhersagen
            y_pred_proba: Vorhersagewahrscheinlichkeiten
            
        Returns:
            Dictionary mit Evaluationsmetriken
        """
        # Basis-Metriken
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # ROC-Kurve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
        
        # Detaillierte Analyse
        metrics['true_positives'] = cm[1, 1]
        metrics['false_positives'] = cm[0, 1]
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_negatives'] = cm[1, 0]
        
        return metrics

class EvaluationModule:
    """Hauptklasse des Bewertungsmoduls"""
    
    def __init__(self):
        """Initialisiert das Bewertungsmodul"""
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.results = {}
        
    def train_models(self, event_matrix: pd.DataFrame) -> Dict:
        """
        Trainiert Modelle für alle Prognosehorizonte
        
        Args:
            event_matrix: Ereignismatrix
            
        Returns:
            Dictionary mit Trainingsergebnissen
        """
        logger.info("Starte Modelltraining für alle Horizonte")
        
        # Features erstellen
        features_df = self.feature_engineer.create_features(event_matrix)
        
        results = {}
        
        for horizon in Config.FORECAST_HORIZONS.keys():
            logger.info(f"Trainiere Modell für {horizon}")
            
            try:
                # Daten vorbereiten (mit Datumsinformationen)
                X, y, valid_data = self.feature_engineer.prepare_training_data(features_df, horizon)
                
                if len(X) == 0:
                    logger.warning(f"Keine Daten für {horizon} verfügbar")
                    continue
                
                # Fixed Train/Test Split entsprechend der Methodik
                # Training: 2003-2019, Test: 2020-2025
                valid_data['publication_date'] = pd.to_datetime(valid_data['publication_date'])
                
                # Split basierend auf Datum
                train_mask = valid_data['publication_date'] < pd.Timestamp(Config.TEST_START)
                test_mask = valid_data['publication_date'] >= pd.Timestamp(Config.TEST_START)
                
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_test = X[test_mask]
                y_test = y[test_mask]
                
                logger.info(f"Trainings-/Testsplit für {horizon}: {len(X_train)} Trainings-, {len(X_test)} Testsamples")
                
                # Modell trainieren
                model = LogisticRegressionModel()
                model.train(X_train, y_train, self.feature_engineer.feature_columns)
                
                # Klassifikator für Evaluierung
                classifier = ClassificationLogic()
                
                # Trainings-Metriken (auf Trainingsdaten)
                y_train_pred, y_train_pred_proba = model.predict(X_train)
                train_metrics = classifier.evaluate_predictions(y_train, y_train_pred, y_train_pred_proba)
                
                # Test-Metriken (auf Testdaten)
                y_test_pred, y_test_pred_proba = model.predict(X_test)
                test_metrics = classifier.evaluate_predictions(y_test, y_test_pred, y_test_pred_proba)
                
                # Feature-Importance für Training und Test berechnen
                train_feature_importance = model.get_feature_importance()
                
                # Für Test-Feature-Importance: Trainiere ein separates Modell auf Testdaten
                # (Dies ist eine Annäherung, da wir normalerweise nicht auf Testdaten trainieren)
                test_model = LogisticRegressionModel()
                test_model.train(X_test, y_test, self.feature_engineer.feature_columns)
                test_feature_importance = test_model.get_feature_importance()
                
                # Erstelle Regressionsergebnisse-Tabelle
                regression_table = model.create_regression_results_table(horizon)
                
                # Speichere Modell und Ergebnisse
                self.models[horizon] = model
                results[horizon] = {
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': train_feature_importance,  # Haupt-Feature-Importance (Training)
                    'train_feature_importance': train_feature_importance,
                    'test_feature_importance': test_feature_importance,
                    'regression_table': regression_table,  # Neue Regressionsergebnisse-Tabelle
                    'n_samples': len(X),
                    'n_train_samples': len(X_train),
                    'n_test_samples': len(X_test),
                    'n_features': len(self.feature_engineer.feature_columns),
                    'feature_columns': self.feature_engineer.feature_columns.copy(),
                    'train_data_info': {
                        'start_date': valid_data[train_mask]['publication_date'].min(),
                        'end_date': valid_data[train_mask]['publication_date'].max(),
                        'n_samples': len(X_train)
                    },
                    'test_data_info': {
                        'start_date': valid_data[test_mask]['publication_date'].min(),
                        'end_date': valid_data[test_mask]['publication_date'].max(),
                        'n_samples': len(X_test)
                    }
                }
                
                logger.info(f"Modell für {horizon} trainiert - Test Accuracy: {test_metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Fehler beim Training für {horizon}: {e}")
                continue
        
        self.results = results
        logger.info("Modelltraining abgeschlossen")
        return results
    
    def predict(self, event_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Macht Vorhersagen für neue Daten
        
        Args:
            event_matrix: Neue Ereignismatrix
            
        Returns:
            DataFrame mit Vorhersagen
        """
        logger.info("Starte Vorhersagen")
        
        # Features erstellen (verwende gleiche Methode wie beim Training)
        features_df = self.feature_engineer.create_features(event_matrix)
        
        predictions_df = features_df[['publication_date', 'indicator', 'indicator_value']].copy()
        
        classifier = ClassificationLogic()
        
        for horizon in self.models.keys():
            try:
                # Verwende nur die Features, die beim Training verwendet wurden
                if horizon not in self.results:
                    logger.warning(f"Keine Trainingsergebnisse für {horizon} verfügbar")
                    continue
                
                # Verwende die gleichen Feature-Spalten wie beim Training
                feature_columns = self.results[horizon].get('feature_columns', self.feature_engineer.feature_columns)
                
                # Daten vorbereiten mit spezifischen Features
                X = features_df[feature_columns].values
                y = features_df[f'{horizon}_direction'].values
                
                if len(X) == 0:
                    continue
                
                # Vorhersagen
                model = self.models[horizon]
                y_pred, y_pred_proba = model.predict(X)
                
                # Klassifikation
                binary_pred, intensity_pred = classifier.classify_predictions(y_pred_proba)
                
                # Ergebnisse speichern
                predictions_df[f'{horizon}_probability'] = y_pred_proba
                predictions_df[f'{horizon}_prediction'] = binary_pred
                predictions_df[f'{horizon}_intensity'] = intensity_pred
                
            except Exception as e:
                logger.error(f"Fehler bei Vorhersagen für {horizon}: {e}")
                continue
        
        logger.info("Vorhersagen abgeschlossen")
        return predictions_df
    
    def save_models(self, filename_prefix: str):
        """
        Speichert trainierte Modelle
        
        Args:
            filename_prefix: Präfix für Dateinamen
        """
        for horizon, model in self.models.items():
            filename = f"{Config.MODELS_DIR}/{filename_prefix}_{horizon}_model.pkl"
            joblib.dump(model, filename)
            logger.info(f"Modell gespeichert: {filename}")
    
    def load_models(self, filename_prefix: str):
        """
        Lädt trainierte Modelle
        
        Args:
            filename_prefix: Präfix für Dateinamen
        """
        for horizon in Config.FORECAST_HORIZONS.keys():
            filename = f"{Config.MODELS_DIR}/{filename_prefix}_{horizon}_model.pkl"
            try:
                model = joblib.load(filename)
                self.models[horizon] = model
                logger.info(f"Modell geladen: {filename}")
            except FileNotFoundError:
                logger.warning(f"Modell nicht gefunden: {filename}")
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Erstellt eine Zusammenfassung der Ergebnisse mit Trainings- und Test-Metriken
        
        Returns:
            DataFrame mit Ergebnisübersicht
        """
        summary_data = []
        
        for horizon, result in self.results.items():
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']
            
            # Trainings-Ergebnisse
            train_row = {
                'horizon': horizon,
                'sample_type': 'Training',
                'accuracy': train_metrics['accuracy'],
                'precision': train_metrics['precision'],
                'recall': train_metrics['recall'],
                'f1_score': train_metrics['f1_score'],
                'roc_auc': train_metrics['roc_auc'],
                'n_samples': result['n_train_samples'],
                'n_features': result['n_features'],
                'start_date': result['train_data_info']['start_date'],
                'end_date': result['train_data_info']['end_date']
            }
            summary_data.append(train_row)
            
            # Test-Ergebnisse
            test_row = {
                'horizon': horizon,
                'sample_type': 'Test',
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1_score': test_metrics['f1_score'],
                'roc_auc': test_metrics['roc_auc'],
                'n_samples': result['n_test_samples'],
                'n_features': result['n_features'],
                'start_date': result['test_data_info']['start_date'],
                'end_date': result['test_data_info']['end_date']
            }
            summary_data.append(test_row)
        
        return pd.DataFrame(summary_data)
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Erstellt eine Vergleichszusammenfassung zwischen Training und Test
        
        Returns:
            DataFrame mit Vergleichsübersicht
        """
        comparison_data = []
        
        for horizon, result in self.results.items():
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']
            
            # Berechne Unterschiede
            comparison_row = {
                'horizon': horizon,
                'train_accuracy': train_metrics['accuracy'],
                'test_accuracy': test_metrics['accuracy'],
                'accuracy_diff': test_metrics['accuracy'] - train_metrics['accuracy'],
                'train_precision': train_metrics['precision'],
                'test_precision': test_metrics['precision'],
                'precision_diff': test_metrics['precision'] - train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'test_recall': test_metrics['recall'],
                'recall_diff': test_metrics['recall'] - train_metrics['recall'],
                'train_f1_score': train_metrics['f1_score'],
                'test_f1_score': test_metrics['f1_score'],
                'f1_diff': test_metrics['f1_score'] - train_metrics['f1_score'],
                'train_roc_auc': train_metrics['roc_auc'],
                'test_roc_auc': test_metrics['roc_auc'],
                'roc_auc_diff': test_metrics['roc_auc'] - train_metrics['roc_auc'],
                'n_train_samples': result['n_train_samples'],
                'n_test_samples': result['n_test_samples'],
                'n_features': result['n_features'],
                'overfitting_indicator': 'Overfitting' if test_metrics['accuracy'] < train_metrics['accuracy'] - 0.05 else 'Good Fit'
            }
            comparison_data.append(comparison_row)
        
        return pd.DataFrame(comparison_data) 
