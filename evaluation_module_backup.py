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
        Erstellt Features für das Modell
        
        Args:
            event_matrix: Ereignismatrix
            
        Returns:
            DataFrame mit Features
        """
        logger.info("Erstelle Features für das Modell")
        
        features_df = event_matrix.copy()
        
        # Basis-Features (standardisierte Indikatorwerte)
        standardized_cols = [col for col in features_df.columns if col.endswith('_standardized')]
        self.feature_columns.extend(standardized_cols)
        
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
        
        # Dummy-Variablen für Zeitmerkmale
        year_dummies = pd.get_dummies(features_df['year'], prefix='year')
        month_dummies = pd.get_dummies(features_df['month'], prefix='month')
        quarter_dummies = pd.get_dummies(features_df['quarter'], prefix='quarter')
        
        features_df = pd.concat([features_df, year_dummies, month_dummies, quarter_dummies], axis=1)
        self.feature_columns.extend(year_dummies.columns.tolist())
        self.feature_columns.extend(month_dummies.columns.tolist())
        self.feature_columns.extend(quarter_dummies.columns.tolist())
        
        # Lag-Features (vorherige Werte)
        for col in standardized_cols:
            features_df[f'{col}_lag1'] = features_df[col].shift(1)
            features_df[f'{col}_lag2'] = features_df[col].shift(2)
            self.feature_columns.extend([f'{col}_lag1', f'{col}_lag2'])
        
        # Rolling-Statistiken
        for col in standardized_cols:
            features_df[f'{col}_rolling_mean_5'] = features_df[col].rolling(window=5).mean()
            features_df[f'{col}_rolling_std_5'] = features_df[col].rolling(window=5).std()
            self.feature_columns.extend([f'{col}_rolling_mean_5', f'{col}_rolling_std_5'])
        
        # Entferne NaN-Werte
        features_df = features_df.dropna()
        
        logger.info(f"Features erstellt: {len(self.feature_columns)} Features")
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
            solver='liblinear'
        )
        self.is_fitted = False
        self.feature_names = []
        
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
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        })
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        return importance_df
    
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
                train_metrics = model.train(X_train, y_train, self.feature_engineer.feature_columns)
                
                # Test-Metriken
                y_pred, y_pred_proba = model.predict(X_test)
                classifier = ClassificationLogic()
                test_metrics = classifier.evaluate_predictions(y_test, y_pred, y_pred_proba)
                
                # Speichere Modell und Ergebnisse
                self.models[horizon] = model
                results[horizon] = {
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': model.get_feature_importance(),
                    'n_samples': len(X),
                    'n_features': len(self.feature_engineer.feature_columns),
                    'feature_columns': self.feature_engineer.feature_columns.copy()
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
        Erstellt eine Zusammenfassung der Ergebnisse
        
        Returns:
            DataFrame mit Ergebnisübersicht
        """
        summary_data = []
        
        for horizon, result in self.results.items():
            test_metrics = result['test_metrics']
            summary_data.append({
                'horizon': horizon,
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1_score': test_metrics['f1_score'],
                'roc_auc': test_metrics['roc_auc'],
                'n_samples': result['n_samples'],
                'n_features': result['n_features']
            })
        
        return pd.DataFrame(summary_data) 