"""
Output- und Evaluationsmodul: Ergebnisaufbereitung, Visualisierung und Validierung
Systematische Aufbereitung, Analyse und Interpretation der Modellprognosen.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import json

from config import Config

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Matplotlib-Styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultProcessor:
    """Klasse zur Ergebnisaufbereitung"""
    
    def __init__(self):
        """Initialisiert den Result Processor"""
        self.results = {}
        
    def process_predictions(self, predictions_df: pd.DataFrame, 
                          event_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Verarbeitet Vorhersagen und erstellt Ergebnisstruktur
        
        Args:
            predictions_df: DataFrame mit Vorhersagen
            event_matrix: Ursprüngliche Ereignismatrix
            
        Returns:
            DataFrame mit vollständigen Ergebnissen
        """
        logger.info("Verarbeite Vorhersageergebnisse")
        
        # Merge mit ursprünglichen Daten
        results_df = predictions_df.merge(
            event_matrix[['publication_date', 'indicator', 'indicator_value'] + 
                        [f'{h}_rate_change' for h in Config.FORECAST_HORIZONS.keys()] +
                        [f'{h}_direction' for h in Config.FORECAST_HORIZONS.keys()]],
            on=['publication_date', 'indicator', 'indicator_value'],
            how='left'
        )
        
        # Füge zusätzliche Informationen hinzu
        results_df['year'] = results_df['publication_date'].dt.year
        results_df['month'] = results_df['publication_date'].dt.month
        results_df['quarter'] = results_df['publication_date'].dt.quarter
        

        
        # Berechne Korrektheit der Prognosen
        for horizon in Config.FORECAST_HORIZONS.keys():
            pred_col = f'{horizon}_prediction'
            actual_col = f'{horizon}_direction'
            
            if pred_col in results_df.columns and actual_col in results_df.columns:
                # Nur für gültige tatsächliche Richtungen
                valid_mask = results_df[actual_col].notna()
                results_df.loc[valid_mask, f'{horizon}_correct'] = (
                    results_df.loc[valid_mask, pred_col] == results_df.loc[valid_mask, actual_col]
                )
        
        # Erstelle Zusammenfassungsstatistiken
        self._create_summary_statistics(results_df)
        
        logger.info("Ergebnisverarbeitung abgeschlossen")
        return results_df
    
    def _create_summary_statistics(self, results_df: pd.DataFrame):
        """
        Erstellt Zusammenfassungsstatistiken
        
        Args:
            results_df: Ergebnisse DataFrame
        """
        summary_stats = {}
        
        for horizon in Config.FORECAST_HORIZONS.keys():
            pred_col = f'{horizon}_prediction'
            prob_col = f'{horizon}_probability'
            intensity_col = f'{horizon}_intensity'
            correct_col = f'{horizon}_correct'
            
            if pred_col in results_df.columns:
                stats = {
                    'total_predictions': len(results_df[results_df[pred_col].notna()]),
                    'long_predictions': (results_df[pred_col] == 1).sum(),
                    'short_predictions': (results_df[pred_col] == 0).sum(),
                    'avg_probability': results_df[prob_col].mean() if prob_col in results_df.columns else None
                }
                
                # Intensitätsverteilung
                if intensity_col in results_df.columns:
                    intensity_dist = results_df[intensity_col].value_counts()
                    stats['intensity_distribution'] = intensity_dist.to_dict()
                
                # Korrektheit
                if correct_col in results_df.columns:
                    correct_predictions = results_df[correct_col].sum()
                    total_valid = results_df[correct_col].notna().sum()
                    stats['accuracy'] = correct_predictions / total_valid if total_valid > 0 else 0
                    stats['correct_predictions'] = correct_predictions
                    stats['total_valid'] = total_valid
                
                summary_stats[horizon] = stats
        
        self.results['summary_statistics'] = summary_stats
        logger.info("Zusammenfassungsstatistiken erstellt")
    
    def create_comprehensive_variable_summary(self, data_df: pd.DataFrame) -> Dict:
        """
        Erstellt umfassende Zusammenfassungsstatistiken für alle Variablen
        
        Args:
            data_df: DataFrame mit allen Variablen
            
        Returns:
            Dictionary mit detaillierten Statistiken für alle Variablen
        """
        logger.info("Erstelle umfassende Variablen-Zusammenfassung")
        
        comprehensive_stats = {}
        variable_docs = Config.get_variable_summary()
        
        # Analysiere jede Variable nach Kategorien
        for category, variables in variable_docs.items():
            category_stats = {}
            
            for var_name, var_info in variables.items():
                # Prüfe ob Variable im DataFrame existiert
                if var_name in data_df.columns:
                    stats = self._analyze_single_variable(data_df[var_name], var_info)
                    category_stats[var_name] = stats
                else:
                    # Prüfe auf Variablen mit Präfix (z.B. für verschiedene Horizonte)
                    matching_cols = [col for col in data_df.columns if col.startswith(var_name.replace('_indicator', ''))]
                    if matching_cols:
                        for col in matching_cols:
                            # Erstelle angepasste Variable-Info
                            adapted_info = var_info.copy()
                            adapted_info['description'] = f"{var_info['description']} ({col})"
                            stats = self._analyze_single_variable(data_df[col], adapted_info)
                            category_stats[col] = stats
            
            comprehensive_stats[category] = category_stats
        
        # Zusätzliche Analysen
        comprehensive_stats['data_overview'] = self._create_data_overview(data_df)
        comprehensive_stats['correlation_analysis'] = self._create_correlation_analysis(data_df)
        comprehensive_stats['missing_data_analysis'] = self._create_missing_data_analysis(data_df)
        comprehensive_stats['temporal_analysis'] = self._create_temporal_analysis(data_df)
        
        self.results['comprehensive_variable_summary'] = comprehensive_stats
        logger.info("Umfassende Variablen-Zusammenfassung erstellt")
        
        return comprehensive_stats
    
    def _analyze_single_variable(self, series: pd.Series, var_info: Dict) -> Dict:
        """
        Analysiert eine einzelne Variable
        
        Args:
            series: Pandas Series mit den Daten
            var_info: Variable-Informationen aus der Konfiguration
            
        Returns:
            Dictionary mit Statistiken
        """
        stats = {
            'variable_info': var_info,
            'basic_stats': {},
            'distribution_stats': {},
            'quality_stats': {},
            'temporal_stats': {}
        }
        
        # Grundlegende Statistiken
        stats['basic_stats'] = {
            'count': series.count(),
            'missing_count': series.isna().sum(),
            'missing_percentage': (series.isna().sum() / len(series)) * 100,
            'unique_values': series.nunique(),
            'data_type': str(series.dtype)
        }
        
        # Numerische Statistiken (falls anwendbar)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_stats = {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis()
            }
            stats['basic_stats'].update(numeric_stats)
            
            # Verteilungsstatistiken (nur für numerische, nicht-boolean Daten)
            if not pd.api.types.is_bool_dtype(series):
                stats['distribution_stats'] = {
                    'percentiles': {
                        'p1': series.quantile(0.01),
                        'p5': series.quantile(0.05),
                        'p10': series.quantile(0.10),
                        'p90': series.quantile(0.90),
                        'p95': series.quantile(0.95),
                        'p99': series.quantile(0.99)
                    },
                    'outliers_iqr': self._detect_outliers_iqr(series),
                    'outliers_zscore': self._detect_outliers_zscore(series)
                }
            else:
                # Boolean-Statistiken
                stats['distribution_stats'] = {
                    'true_count': series.sum(),
                    'false_count': (~series).sum(),
                    'true_percentage': (series.sum() / len(series)) * 100,
                    'false_percentage': ((~series).sum() / len(series)) * 100
                }
        
        # Kategorische Statistiken (falls anwendbar)
        if pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
            value_counts = series.value_counts()
            stats['distribution_stats'] = {
                'value_counts': value_counts.to_dict(),
                'top_categories': value_counts.head(10).to_dict(),
                'category_distribution': (value_counts / len(series) * 100).to_dict()
            }
        
        # Datumsstatistiken (falls anwendbar)
        if pd.api.types.is_datetime64_any_dtype(series):
            stats['temporal_stats'] = {
                'date_range': {
                    'start': series.min(),
                    'end': series.max(),
                    'duration_days': (series.max() - series.min()).days
                },
                'year_distribution': series.dt.year.value_counts().to_dict(),
                'month_distribution': series.dt.month.value_counts().to_dict(),
                'day_of_week_distribution': series.dt.dayofweek.value_counts().to_dict()
            }
        
        # Qualitätsstatistiken
        stats['quality_stats'] = {
            'completeness': (series.count() / len(series)) * 100,
            'consistency': self._check_consistency(series, var_info),
            'expected_range_check': self._check_expected_range(series, var_info),
            'data_quality_score': self._calculate_quality_score(series, var_info)
        }
        
        return stats
    
    def _detect_outliers_iqr(self, series: pd.Series) -> Dict:
        """Erkennt Ausreißer mit IQR-Methode"""
        if pd.api.types.is_bool_dtype(series):
            return {
                'outlier_count': 0,
                'outlier_percentage': 0.0,
                'lower_bound': None,
                'upper_bound': None,
                'outlier_values': [],
                'note': 'IQR-Methode nicht anwendbar für Boolean-Daten'
            }
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_values': outliers.tolist() if len(outliers) <= 10 else outliers.head(10).tolist()
        }
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> Dict:
        """Erkennt Ausreißer mit Z-Score-Methode"""
        if pd.api.types.is_bool_dtype(series):
            return {
                'outlier_count': 0,
                'outlier_percentage': 0.0,
                'threshold': threshold,
                'outlier_values': [],
                'note': 'Z-Score-Methode nicht anwendbar für Boolean-Daten'
            }
        
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > threshold]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series)) * 100,
            'threshold': threshold,
            'outlier_values': outliers.tolist() if len(outliers) <= 10 else outliers.head(10).tolist()
        }
    
    def _check_consistency(self, series: pd.Series, var_info: Dict) -> Dict:
        """Prüft Konsistenz der Daten"""
        checks = {
            'data_type_consistent': True,
            'range_consistent': True,
            'format_consistent': True
        }
        
        # Prüfe Datentyp-Konsistenz
        if 'data_type' in var_info:
            expected_type = var_info['data_type']
            if 'float' in expected_type and not pd.api.types.is_float_dtype(series):
                checks['data_type_consistent'] = False
            elif 'int' in expected_type and not pd.api.types.is_integer_dtype(series):
                checks['data_type_consistent'] = False
        
        # Prüfe Bereich-Konsistenz
        if 'expected_range' in var_info and pd.api.types.is_numeric_dtype(series):
            range_str = var_info['expected_range']
            if 'bis' in range_str:
                try:
                    parts = range_str.split('bis')
                    min_val = float(parts[0].strip().split()[-1])
                    max_val = float(parts[1].strip().split()[0])
                    
                    if series.min() < min_val or series.max() > max_val:
                        checks['range_consistent'] = False
                except:
                    pass
        
        return checks
    
    def _check_expected_range(self, series: pd.Series, var_info: Dict) -> Dict:
        """Prüft ob Werte im erwarteten Bereich liegen"""
        if not pd.api.types.is_numeric_dtype(series):
            return {'applicable': False}
        
        range_check = {
            'applicable': True,
            'in_range_count': 0,
            'out_of_range_count': 0,
            'range_violations': []
        }
        
        if 'expected_range' in var_info:
            range_str = var_info['expected_range']
            if 'bis' in range_str:
                try:
                    parts = range_str.split('bis')
                    min_val = float(parts[0].strip().split()[-1])
                    max_val = float(parts[1].strip().split()[0])
                    
                    in_range = (series >= min_val) & (series <= max_val)
                    range_check['in_range_count'] = in_range.sum()
                    range_check['out_of_range_count'] = (~in_range).sum()
                    range_check['in_range_percentage'] = (in_range.sum() / len(series)) * 100
                    
                    # Sammle Verletzungen
                    violations = series[~in_range]
                    if len(violations) > 0:
                        range_check['range_violations'] = violations.head(10).tolist()
                        
                except:
                    range_check['applicable'] = False
        
        return range_check
    
    def _calculate_quality_score(self, series: pd.Series, var_info: Dict) -> float:
        """Berechnet einen Datenqualitäts-Score"""
        score = 0.0
        max_score = 100.0
        
        # Vollständigkeit (40 Punkte)
        completeness = (series.count() / len(series)) * 100
        score += (completeness / 100) * 40
        
        # Konsistenz (30 Punkte)
        consistency_checks = self._check_consistency(series, var_info)
        consistency_score = sum(consistency_checks.values()) / len(consistency_checks) * 100
        score += (consistency_score / 100) * 30
        
        # Bereich-Konsistenz (30 Punkte)
        range_check = self._check_expected_range(series, var_info)
        if range_check['applicable']:
            range_score = range_check.get('in_range_percentage', 100)
            score += (range_score / 100) * 30
        else:
            score += 30  # Nicht anwendbar = voller Score
        
        return min(score, max_score)
    
    def _create_data_overview(self, data_df: pd.DataFrame) -> Dict:
        """Erstellt eine Übersicht über den gesamten Datensatz"""
        return {
            'total_rows': len(data_df),
            'total_columns': len(data_df.columns),
            'memory_usage_mb': data_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': {
                'start': data_df['publication_date'].min() if 'publication_date' in data_df.columns else None,
                'end': data_df['publication_date'].max() if 'publication_date' in data_df.columns else None
            },
            'column_types': data_df.dtypes.to_dict(),
            'missing_data_summary': data_df.isnull().sum().to_dict()
        }
    
    def _create_correlation_analysis(self, data_df: pd.DataFrame) -> Dict:
        """Erstellt Korrelationsanalyse für numerische Variablen"""
        numeric_cols = data_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            correlation_matrix = data_df[numeric_cols].corr()
            
            # Finde starke Korrelationen
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Starke Korrelation
                        strong_correlations.append({
                            'variable1': correlation_matrix.columns[i],
                            'variable2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'highest_correlations': sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)[:10]
            }
        else:
            return {'message': 'Nicht genügend numerische Variablen für Korrelationsanalyse'}
    
    def _create_missing_data_analysis(self, data_df: pd.DataFrame) -> Dict:
        """Analysiert fehlende Daten"""
        missing_data = data_df.isnull().sum()
        missing_percentage = (missing_data / len(data_df)) * 100
        
        return {
            'missing_counts': missing_data.to_dict(),
            'missing_percentages': missing_percentage.to_dict(),
            'variables_with_missing_data': missing_data[missing_data > 0].to_dict(),
            'variables_without_missing_data': missing_data[missing_data == 0].to_dict(),
            'total_missing_cells': missing_data.sum(),
            'overall_completeness': ((len(data_df) * len(data_df.columns) - missing_data.sum()) / 
                                   (len(data_df) * len(data_df.columns))) * 100
        }
    
    def _create_temporal_analysis(self, data_df: pd.DataFrame) -> Dict:
        """Erstellt zeitliche Analyse"""
        if 'publication_date' not in data_df.columns:
            return {'message': 'Keine Zeitstempel-Spalte gefunden'}
        
        temporal_stats = {
            'date_range': {
                'start': data_df['publication_date'].min(),
                'end': data_df['publication_date'].max(),
                'duration_days': (data_df['publication_date'].max() - data_df['publication_date'].min()).days
            },
            'yearly_distribution': data_df['publication_date'].dt.year.value_counts().sort_index().to_dict(),
            'monthly_distribution': data_df['publication_date'].dt.month.value_counts().sort_index().to_dict(),
            'quarterly_distribution': data_df['publication_date'].dt.quarter.value_counts().sort_index().to_dict(),
            'daily_distribution': data_df['publication_date'].dt.dayofweek.value_counts().sort_index().to_dict()
        }
        
        # Analysiere Datenlücken
        date_range = pd.date_range(start=data_df['publication_date'].min(), 
                                 end=data_df['publication_date'].max(), 
                                 freq='D')
        missing_dates = date_range.difference(data_df['publication_date'].dt.date)
        
        temporal_stats['data_gaps'] = {
            'total_missing_dates': len(missing_dates),
            'missing_dates_percentage': (len(missing_dates) / len(date_range)) * 100,
            'largest_gap_days': self._find_largest_gap(data_df['publication_date'])
        }
        
        return temporal_stats
    
    def _find_largest_gap(self, date_series: pd.Series) -> int:
        """Findet die größte Lücke zwischen aufeinanderfolgenden Daten"""
        sorted_dates = date_series.sort_values()
        gaps = (sorted_dates - sorted_dates.shift(1)).dt.days
        return gaps.max() if len(gaps) > 0 else 0

class Visualizer:
    """Klasse zur Visualisierung der Ergebnisse"""
    
    def __init__(self):
        """Initialisiert den Visualizer"""
        self.figure_size = Config.FIGURE_SIZE
        self.dpi = Config.DPI
        
        # Feature-Name-Übersetzungen (FRED-Code -> Deutsche Bezeichnung)
        self.feature_translations = {
            # Basis-Indikatoren
            'GDP_standardized': 'BIP USA',
            'CLVMNACSCAB1GQEU28_standardized': 'BIP Eurozone',
            'CPIAUCSL_standardized': 'CPI USA',
            'CP0000EZ19M086NEST_standardized': 'HVPI Eurozone',
            'PPIACO_standardized': 'PPI USA',
            'PIEAMP01EUM661N_standardized': 'PPI Eurozone',
            'FEDFUNDS_standardized': 'Fed Funds Rate',
            'IR3TIB01EZM156N_standardized': '3M EURIBOR',
            'UNRATE_standardized': 'Arbeitslosenquote USA',
            'LRUNTTTTDEQ156S_standardized': 'Arbeitslosenquote DE',
            'UMCSENT_standardized': 'Verbrauchervertrauen USA',
            'BSCICP03EZM665S_standardized': 'Verbrauchervertrauen EU',
            
            # Abgeleitete Indikatoren
            'UNRATE_diff_standardized': 'US-Arbeitslosigkeit (Änderung)',
            'LRUNTTTTDEQ156S_diff_standardized': 'DE-Arbeitslosigkeit (Änderung)',
            'INTEREST_RATE_SPREAD_standardized': 'Zinsspread (USA-EU)',
            'GDP_growth_standardized': 'US-BIP-Wachstum',
            'CLVMNACSCAB1GQEU28_growth_standardized': 'EU-BIP-Wachstum',
            'CPIAUCSL_yoy_standardized': 'US-Inflation (YoY)',
            'CP0000EZ19M086NEST_yoy_standardized': 'EU-Inflation (YoY)',
            'PPIACO_yoy_standardized': 'US-PPI (YoY)',
            'PIEAMP01EUM661N_yoy_standardized': 'EU-PPI (YoY)',
        }
    
    def translate_feature_name(self, feature_name: str) -> str:
        """
        Übersetzt Feature-Namen von FRED-Codes zu deutschen Bezeichnungen
        
        Args:
            feature_name: Original Feature-Name
            
        Returns:
            Deutsche Bezeichnung oder Original-Name falls keine Übersetzung verfügbar
        """
        # Direkte Übersetzung
        if feature_name in self.feature_translations:
            return self.feature_translations[feature_name]
        
        # Lag-Features
        if '_lag1' in feature_name:
            base_name = feature_name.replace('_lag1', '')
            if base_name in self.feature_translations:
                return f"{self.feature_translations[base_name]} (Lag-1)"
        
        if '_lag2' in feature_name:
            base_name = feature_name.replace('_lag2', '')
            if base_name in self.feature_translations:
                return f"{self.feature_translations[base_name]} (Lag-2)"
        
        # Rolling-Features
        if '_rolling_mean_5' in feature_name:
            base_name = feature_name.replace('_rolling_mean_5', '')
            if base_name in self.feature_translations:
                return f"{self.feature_translations[base_name]} (5-Periode Ø)"
        
        if '_rolling_std_5' in feature_name:
            base_name = feature_name.replace('_rolling_std_5', '')
            if base_name in self.feature_translations:
                return f"{self.feature_translations[base_name]} (5-Periode σ)"
        
        # Indikator-Dummies
        if feature_name.startswith('indicator_'):
            indicator_code = feature_name.replace('indicator_', '')
            if f"{indicator_code}_standardized" in self.feature_translations:
                return f"Indikator: {self.feature_translations[f'{indicator_code}_standardized']}"
            # Spezialfall für EURIBOR
            elif indicator_code == 'IR3TIB01EZM156N':
                return f"Indikator: 3M EURIBOR"
        
        # Zeit-Features
        time_translations = {
            'year': 'Jahr',
            'month': 'Monat', 
            'quarter': 'Quartal',
            'day_of_week': 'Wochentag'
        }
        
        for time_key, time_name in time_translations.items():
            if feature_name.startswith(f'{time_key}_'):
                return f"{time_name}: {feature_name.split('_', 1)[1]}"
        
        # Falls keine Übersetzung gefunden, gib Original zurück
        return feature_name
        
    def plot_time_series(self, results_df: pd.DataFrame, horizon: str, 
                        save_path: Optional[str] = None):
        """
        Erstellt Zeitreihenplots von Prognosen und Ist-Kursen
        
        Args:
            results_df: Ergebnisse DataFrame
            horizon: Prognosehorizont
            save_path: Speicherpfad (optional)
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figure_size, dpi=self.dpi)
        
        # Plot 1: Wahrscheinlichkeiten über Zeit
        prob_col = f'{horizon}_probability'
        if prob_col in results_df.columns:
            axes[0].plot(results_df['publication_date'], results_df[prob_col], 
                        alpha=0.7, linewidth=1)
            axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Schwellenwert')
            axes[0].set_title(f'Wahrscheinlichkeiten für {horizon} Horizont')
            axes[0].set_ylabel('Wahrscheinlichkeit')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Kursänderungen über Zeit
        rate_change_col = f'{horizon}_rate_change'
        if rate_change_col in results_df.columns:
            axes[1].plot(results_df['publication_date'], results_df[rate_change_col], 
                        alpha=0.7, linewidth=1)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1].axhline(y=Config.MOVEMENT_THRESHOLD, color='green', linestyle='--', alpha=0.5, label='Long-Schwelle')
            axes[1].axhline(y=-Config.MOVEMENT_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='Short-Schwelle')
            axes[1].set_title(f'Tatsächliche Kursänderungen für {horizon} Horizont')
            axes[1].set_ylabel('Kursänderung (%)')
            axes[1].set_xlabel('Datum')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Zeitreihenplot gespeichert: {save_path}")
        
        plt.close()
    
    def plot_roc_curves(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """
        Erstellt ROC-Kurven für alle Horizonte
        
        Args:
            evaluation_results: Evaluationsergebnisse
            save_path: Speicherpfad (optional)
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(evaluation_results)))
        
        for i, (horizon, result) in enumerate(evaluation_results.items()):
            if 'roc_curve' in result['test_metrics']:
                roc_data = result['test_metrics']['roc_curve']
                auc = result['test_metrics']['roc_auc']
                
                ax.plot(roc_data['fpr'], roc_data['tpr'], 
                       color=colors[i], linewidth=2, 
                       label=f'{horizon} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Zufällig')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC-Kurven für alle Prognosehorizonte')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Füge AUC-Werte als Text hinzu
        for i, (horizon, result) in enumerate(evaluation_results.items()):
            if 'roc_auc' in result['test_metrics']:
                auc = result['test_metrics']['roc_auc']
                ax.text(0.6, 0.1 + i*0.1, f'{horizon}: AUC = {auc:.3f}', 
                       transform=ax.transAxes, fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ROC-Kurven gespeichert: {save_path}")
        
        plt.close()
    
    def plot_training_test_comparison(self, results_summary: pd.DataFrame, save_path: Optional[str] = None):
        """
        Erstellt Vergleichsplots zwischen Training und Test für alle Metriken
        
        Args:
            results_summary: DataFrame mit Training und Test Metriken
            save_path: Speicherpfad (optional)
        """
        # Erstelle Subplots für verschiedene Metriken
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                
                # Gruppiere Daten nach Horizont
                horizons = results_summary['horizon'].unique()
                x_pos = np.arange(len(horizons))
                width = 0.35
                
                train_values = []
                test_values = []
                
                for horizon in horizons:
                    train_data = results_summary[(results_summary['horizon'] == horizon) & 
                                               (results_summary['sample_type'] == 'Training')]
                    test_data = results_summary[(results_summary['horizon'] == horizon) & 
                                              (results_summary['sample_type'] == 'Test')]
                    
                    if not train_data.empty:
                        train_values.append(train_data[metric].iloc[0])
                    else:
                        train_values.append(0)
                        
                    if not test_data.empty:
                        test_values.append(test_data[metric].iloc[0])
                    else:
                        test_values.append(0)
                
                # Erstelle Balkendiagramm
                bars1 = ax.bar(x_pos - width/2, train_values, width, label='Training', 
                              color='skyblue', alpha=0.8)
                bars2 = ax.bar(x_pos + width/2, test_values, width, label='Test', 
                              color='lightcoral', alpha=0.8)
                
                # Füge Werte über den Balken hinzu
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                
                ax.set_xlabel('Horizont')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} - Training vs Test')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(horizons)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Setze y-Achse von 0 bis 1 für Klassifikationsmetriken
                if metric != 'roc_auc':
                    ax.set_ylim(0, 1)
        
        # Entferne das letzte leere Subplot
        if len(axes) > len(metrics):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training vs Test Vergleich gespeichert: {save_path}")
        
        plt.close()
    
    def plot_overfitting_analysis(self, comparison_summary: pd.DataFrame, save_path: Optional[str] = None):
        """
        Erstellt Overfitting-Analyse Visualisierung
        
        Args:
            comparison_summary: DataFrame mit Vergleichsdaten
            save_path: Speicherpfad (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        
        # Plot 1: Accuracy-Differenzen
        ax1 = axes[0, 0]
        horizons = comparison_summary['horizon']
        accuracy_diffs = comparison_summary['accuracy_diff']
        colors = ['red' if diff < -0.05 else 'green' if diff > 0.05 else 'orange' for diff in accuracy_diffs]
        
        bars = ax1.bar(horizons, accuracy_diffs, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5, label='Overfitting-Schwelle')
        ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Underfitting-Schwelle')
        
        # Füge Werte über den Balken hinzu
        for bar, diff in zip(bars, accuracy_diffs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{diff:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        ax1.set_title('Accuracy-Differenz (Test - Training)')
        ax1.set_ylabel('Differenz')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1-Score-Differenzen
        ax2 = axes[0, 1]
        f1_diffs = comparison_summary['f1_diff']
        colors = ['red' if diff < -0.05 else 'green' if diff > 0.05 else 'orange' for diff in f1_diffs]
        
        bars = ax2.bar(horizons, f1_diffs, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.5)
        
        for bar, diff in zip(bars, f1_diffs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{diff:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        ax2.set_title('F1-Score-Differenz (Test - Training)')
        ax2.set_ylabel('Differenz')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sample-Größen Vergleich
        ax3 = axes[1, 0]
        train_samples = comparison_summary['n_train_samples']
        test_samples = comparison_summary['n_test_samples']
        
        x_pos = np.arange(len(horizons))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, train_samples, width, label='Training', color='skyblue', alpha=0.8)
        bars2 = ax3.bar(x_pos + width/2, test_samples, width, label='Test', color='lightcoral', alpha=0.8)
        
        for bar in bars1:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_xlabel('Horizont')
        ax3.set_ylabel('Anzahl Samples')
        ax3.set_title('Sample-Größen - Training vs Test')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(horizons)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overfitting-Status
        ax4 = axes[1, 1]
        overfitting_status = comparison_summary['overfitting_indicator']
        status_counts = overfitting_status.value_counts()
        
        colors = ['green' if status == 'Good Fit' else 'red' for status in status_counts.index]
        wedges, texts, autotexts = ax4.pie(status_counts.values, labels=status_counts.index, 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        
        ax4.set_title('Overfitting-Status Verteilung')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Overfitting-Analyse gespeichert: {save_path}")
        
        plt.close()
    
    def plot_metrics_heatmap(self, results_summary: pd.DataFrame, save_path: Optional[str] = None):
        """
        Erstellt Heatmap der Metriken für alle Horizonte
        
        Args:
            results_summary: DataFrame mit Training und Test Metriken
            save_path: Speicherpfad (optional)
        """
        # Metriken für Heatmap
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Erstelle separate Heatmaps für Training und Test
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=self.dpi)
        
        for idx, sample_type in enumerate(['Training', 'Test']):
            ax = axes[idx]
            
            # Filtere Daten nach Sample-Typ
            sample_data = results_summary[results_summary['sample_type'] == sample_type]
            
            if not sample_data.empty:
                # Erstelle Heatmap-Daten
                heatmap_data = sample_data.set_index('horizon')[metrics]
                
                # Erstelle Heatmap
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                           center=0.5, ax=ax, cbar_kws={'label': 'Metrik-Wert'})
                
                ax.set_title(f'{sample_type} Metriken Heatmap')
                ax.set_xlabel('Metriken')
                ax.set_ylabel('Horizont')
            else:
                ax.text(0.5, 0.5, f'Keine {sample_type} Daten verfügbar', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{sample_type} Metriken Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Metriken Heatmap gespeichert: {save_path}")
        
        plt.close()
    
    def plot_feature_importance_comparison(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """
        Erstellt Feature-Importance Vergleich zwischen Training und Test
        
        Args:
            evaluation_results: Dictionary mit Evaluationsergebnissen
            save_path: Speicherpfad (optional)
        """
        # Erstelle Subplots für jeden Horizont
        horizons = list(evaluation_results.keys())
        n_horizons = len(horizons)
        
        # Berechne Subplot-Layout
        cols = min(2, n_horizons)
        rows = (n_horizons + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows), dpi=self.dpi)
        if n_horizons == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        for idx, horizon in enumerate(horizons):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            result = evaluation_results[horizon]
            
            # Feature-Importance Daten extrahieren
            train_importance = result.get('train_feature_importance', pd.DataFrame())
            test_importance = result.get('test_feature_importance', pd.DataFrame())

            # Entferne generischen Platzhalter 'indicator_value_standardized'
            if not train_importance.empty:
                train_importance = train_importance[train_importance['feature'] != 'indicator_value_standardized']
            if not test_importance.empty:
                test_importance = test_importance[test_importance['feature'] != 'indicator_value_standardized']

            # Nach dem Filtern prüfen, ob noch Daten vorhanden sind
            if train_importance.empty or test_importance.empty:
                ax.text(0.5, 0.5, f'Keine Feature-Importance Daten für {horizon}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Feature-Importance: {horizon}')
                continue
            
            # Nutze alle Features, damit keine Indikatoren fehlen
            top_features = train_importance['feature'].tolist()
            
            # Filtere Daten für Top Features
            train_top = train_importance[train_importance['feature'].isin(top_features)]
            test_top = test_importance[test_importance['feature'].isin(top_features)]
            
            # Sortiere nach Training-Importance
            train_top = train_top.sort_values('abs_coefficient', ascending=True)
            
            # Erstelle horizontale Balkendiagramme
            y_pos = np.arange(len(train_top))
            width = 0.35
            
            # Training-Koeffizienten
            train_bars = ax.barh(y_pos - width/2, train_top['coefficient'], width, 
                               label='Training', color='skyblue', alpha=0.8)
            
            # Test-Koeffizienten
            test_bars = ax.barh(y_pos + width/2, test_top['coefficient'], width, 
                              label='Test', color='lightcoral', alpha=0.8)
            
            # Y-Achse Labels (Feature-Namen) - Übersetze zu deutschen Bezeichnungen
            ax.set_yticks(y_pos)
            translated_labels = [self.translate_feature_name(feature) for feature in train_top['feature']]
            ax.set_yticklabels(translated_labels, fontsize=9)
            
            # X-Achse
            ax.set_xlabel('Koeffizient-Wert')
            ax.set_title(f'Feature-Importance Vergleich: {horizon}')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
            
            # Vertikale Linie bei 0
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Füge Koeffizient-Werte hinzu
            for i, (train_coef, test_coef) in enumerate(zip(train_top['coefficient'], test_top['coefficient'])):
                # Training-Wert
                ax.text(train_coef + (0.01 if train_coef >= 0 else -0.01), 
                       y_pos[i] - width/2, f'{train_coef:.3f}', 
                       ha='left' if train_coef >= 0 else 'right', va='center', fontsize=8)
                
                # Test-Wert
                ax.text(test_coef + (0.01 if test_coef >= 0 else -0.01), 
                       y_pos[i] + width/2, f'{test_coef:.3f}', 
                       ha='left' if test_coef >= 0 else 'right', va='center', fontsize=8)
        
        # Entferne leere Subplots
        for idx in range(n_horizons, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature-Importance Vergleich gespeichert: {save_path}")
        
        plt.close()
    
    def plot_feature_importance_heatmap(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """
        Erstellt Heatmap der Feature-Importance für alle Horizonte
        
        Args:
            evaluation_results: Dictionary mit Evaluationsergebnissen
            save_path: Speicherpfad (optional)
        """
        # Sammle alle Feature-Importance Daten
        all_features = set()
        horizon_data = {}
        
        for horizon, result in evaluation_results.items():
            train_importance = result.get('train_feature_importance', pd.DataFrame())
            test_importance = result.get('test_feature_importance', pd.DataFrame())

            # Entferne generischen Platzhalter 'indicator_value_standardized'
            if not train_importance.empty:
                train_importance = train_importance[train_importance['feature'] != 'indicator_value_standardized']
            if not test_importance.empty:
                test_importance = test_importance[test_importance['feature'] != 'indicator_value_standardized']

            if not train_importance.empty:
                all_features.update(train_importance['feature'].tolist())
            if not test_importance.empty:
                all_features.update(test_importance['feature'].tolist())
            
            horizon_data[horizon] = {
                'train': train_importance,
                'test': test_importance
            }
        
        if not all_features:
            logger.warning("Keine Feature-Importance Daten verfügbar")
            return
        
        # Erstelle Subplots für Training und Test
        fig, axes = plt.subplots(2, 1, figsize=(16, 12), dpi=self.dpi)
        
        # Top 20 Features für bessere Übersichtlichkeit
        top_features = sorted(list(all_features))[:20]
        
        for idx, sample_type in enumerate(['train', 'test']):
            ax = axes[idx]
            
            # Erstelle Heatmap-Daten
            heatmap_data = []
            feature_names = []
            
            for feature in top_features:
                feature_row = []
                for horizon in evaluation_results.keys():
                    importance_data = horizon_data[horizon][sample_type]
                    if not importance_data.empty:
                        feature_data = importance_data[importance_data['feature'] == feature]
                        if not feature_data.empty:
                            feature_row.append(feature_data.iloc[0]['coefficient'])
                        else:
                            feature_row.append(0)
                    else:
                        feature_row.append(0)
                heatmap_data.append(feature_row)
                feature_names.append(self.translate_feature_name(feature))
            
            if heatmap_data:
                # Erstelle Heatmap
                heatmap_array = np.array(heatmap_data)
                im = ax.imshow(heatmap_array, cmap='RdBu_r', aspect='auto', 
                             vmin=-np.max(np.abs(heatmap_array)), vmax=np.max(np.abs(heatmap_array)))
                
                # Achsen-Labels
                ax.set_xticks(range(len(evaluation_results.keys())))
                ax.set_xticklabels(evaluation_results.keys())
                ax.set_yticks(range(len(feature_names)))
                ax.set_yticklabels(feature_names, fontsize=9)
                
                # Farbbalken
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Koeffizient-Wert')
                
                # Werte in Zellen
                for i in range(len(feature_names)):
                    for j in range(len(evaluation_results.keys())):
                        value = heatmap_array[i, j]
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                               fontsize=8, color='black' if abs(value) < 0.5 else 'white')
                
                ax.set_title(f'Feature-Importance Heatmap: {sample_type.title()}')
                ax.set_xlabel('Horizont')
                ax.set_ylabel('Features')
            else:
                ax.text(0.5, 0.5, f'Keine {sample_type} Feature-Importance Daten verfügbar', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'Feature-Importance Heatmap: {sample_type.title()}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature-Importance Heatmap gespeichert: {save_path}")
        
        plt.close()

    def plot_feature_importance_training_only(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """
        Erstellt einen Plot der Feature-Importance nur aus den Trainingskoeffizienten
        für alle Horizonte. Blendet 'indicator_value_standardized' aus.
        
        Args:
            evaluation_results: Dictionary mit Evaluationsergebnissen
            save_path: Speicherpfad (optional)
        """
        horizons = list(evaluation_results.keys())
        n_horizons = len(horizons)
        
        cols = min(2, n_horizons)
        rows = (n_horizons + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows), dpi=self.dpi)
        if n_horizons == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        for idx, horizon in enumerate(horizons):
            if idx >= len(axes):
                break
            ax = axes[idx]
            result = evaluation_results[horizon]
            
            train_importance = result.get('train_feature_importance', pd.DataFrame())
            
            # Filtere unerwünschtes generisches Feature
            if not train_importance.empty:
                train_importance = train_importance[train_importance['feature'] != 'indicator_value_standardized']
            
            if train_importance.empty:
                ax.text(0.5, 0.5, f'Keine Trainings-Feature-Importance für {horizon}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Train Feature-Importance: {horizon}')
                continue
            
            # Zeige alle Features (keine Trunkierung), damit keine Indikatoren fehlen
            train_top = train_importance.sort_values('abs_coefficient', ascending=False)
            train_top = train_top.sort_values('abs_coefficient', ascending=True)
            
            y_pos = np.arange(len(train_top))
            bars = ax.barh(y_pos, train_top['coefficient'], color='steelblue', alpha=0.85)
            
            # Übersetzte Labels
            ax.set_yticks(y_pos)
            translated_labels = [self.translate_feature_name(feature) for feature in train_top['feature']]
            ax.set_yticklabels(translated_labels, fontsize=8)
            
            ax.set_xlabel('Koeffizient-Wert')
            ax.set_title(f'Train Feature-Importance: {horizon}')
            ax.grid(True, alpha=0.3, axis='x')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Werte an Balken: alle Balken beschriften
            x_min, x_max = ax.get_xlim()
            x_range = x_max - x_min if x_max > x_min else 1.0
            inner_pad = 0.02 * x_range
            inner_threshold = 0.03 * x_range  # ab dieser Breite Text innerhalb
            for i, (bar, coef) in enumerate(zip(bars, train_top['coefficient'])):
                if abs(coef) >= inner_threshold:
                    # Beschriftung innerhalb des Balkens (weiße Schrift)
                    x_pos = coef - (inner_pad if coef >= 0 else -inner_pad)
                    ha = 'right' if coef >= 0 else 'left'
                    color = 'white'
                else:
                    # Beschriftung außerhalb des Balkens (schwarze Schrift)
                    x_pos = coef + (inner_pad if coef >= 0 else -inner_pad)
                    ha = 'left' if coef >= 0 else 'right'
                    color = 'black'
                txt = ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                              f'{coef:.3f}', ha=ha, va='center', fontsize=7, color=color)
                # Schwarze Kontur für bessere Lesbarkeit
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=1.0, foreground='black'),
                    path_effects.Normal()
                ])
        
        # Entferne überzählige Subplots
        for idx in range(n_horizons, len(axes)):
            fig.delaxes(axes[idx])
        
        # Mehr Platz für Y-Labels links
        plt.tight_layout()
        try:
            plt.subplots_adjust(left=0.38)
        except Exception:
            pass
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Train Feature-Importance gespeichert: {save_path}")
        plt.close()

    def plot_feature_importance_test_only(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """
        Plot der Feature-Importance nur aus Test-Koeffizienten für alle Horizonte.
        Blendet 'indicator_value_standardized' aus und zeigt alle Features.
        """
        horizons = list(evaluation_results.keys())
        n_horizons = len(horizons)

        cols = min(2, n_horizons)
        rows = (n_horizons + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows), dpi=self.dpi)
        if n_horizons == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()

        for idx, horizon in enumerate(horizons):
            if idx >= len(axes):
                break
            ax = axes[idx]
            result = evaluation_results[horizon]

            test_importance = result.get('test_feature_importance', pd.DataFrame())
            if not test_importance.empty:
                test_importance = test_importance[test_importance['feature'] != 'indicator_value_standardized']

            if test_importance.empty:
                ax.text(0.5, 0.5, f'Keine Test-Feature-Importance für {horizon}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Test Feature-Importance: {horizon}')
                continue

            test_top = test_importance.sort_values('abs_coefficient', ascending=False)
            test_top = test_top.sort_values('abs_coefficient', ascending=True)

            y_pos = np.arange(len(test_top))
            bars = ax.barh(y_pos, test_top['coefficient'], color='indianred', alpha=0.85)

            ax.set_yticks(y_pos)
            translated_labels = [self.translate_feature_name(feature) for feature in test_top['feature']]
            ax.set_yticklabels(translated_labels, fontsize=8)

            ax.set_xlabel('Koeffizient-Wert')
            ax.set_title(f'Test Feature-Importance: {horizon}')
            ax.grid(True, alpha=0.3, axis='x')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

            x_min, x_max = ax.get_xlim()
            x_range = x_max - x_min if x_max > x_min else 1.0
            inner_pad = 0.02 * x_range
            inner_threshold = 0.03 * x_range
            for i, (bar, coef) in enumerate(zip(bars, test_top['coefficient'])):
                if abs(coef) >= inner_threshold:
                    x_pos = coef - (inner_pad if coef >= 0 else -inner_pad)
                    ha = 'right' if coef >= 0 else 'left'
                    color = 'white'
                else:
                    x_pos = coef + (inner_pad if coef >= 0 else -inner_pad)
                    ha = 'left' if coef >= 0 else 'right'
                    color = 'black'
                txt = ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                              f'{coef:.3f}', ha=ha, va='center', fontsize=7, color=color)
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=1.0, foreground='black'),
                    path_effects.Normal()
                ])

        for idx in range(n_horizons, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        try:
            plt.subplots_adjust(left=0.38)
        except Exception:
            pass
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Test Feature-Importance gespeichert: {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """
        Erstellt Confusion Matrizen für Training und Test für alle Horizonte
        
        Args:
            evaluation_results: Evaluationsergebnisse
            save_path: Speicherpfad (optional)
        """
        n_horizons = len(evaluation_results)
        
        # Erstelle Subplots: 2 Reihen (Training, Test) x Anzahl Horizonte
        fig, axes = plt.subplots(2, n_horizons, figsize=(5*n_horizons, 10), dpi=self.dpi)
        
        # Falls nur ein Horizont, mache axes zu 2D-Array
        if n_horizons == 1:
            axes = axes.reshape(2, 1)
        
        for i, (horizon, result) in enumerate(evaluation_results.items()):
            if i >= n_horizons:
                break
            
            # Training Confusion Matrix
            train_cm = result['train_metrics']['confusion_matrix']
            sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Short', 'Long'], 
                       yticklabels=['Short', 'Long'],
                       ax=axes[0, i])
            
            axes[0, i].set_title(f'Training - {horizon} Horizont')
            axes[0, i].set_xlabel('Vorhersage')
            axes[0, i].set_ylabel('Tatsächlich')
            
            # Test Confusion Matrix
            test_cm = result['test_metrics']['confusion_matrix']
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Short', 'Long'], 
                       yticklabels=['Short', 'Long'],
                       ax=axes[1, i])
            
            axes[1, i].set_title(f'Test - {horizon} Horizont')
            axes[1, i].set_xlabel('Vorhersage')
            axes[1, i].set_ylabel('Tatsächlich')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training vs Test Confusion Matrizen gespeichert: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix_metrics_comparison(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """
        Erstellt Vergleich der Confusion Matrix Metriken zwischen Training und Test
        
        Args:
            evaluation_results: Evaluationsergebnisse
            save_path: Speicherpfad (optional)
        """
        # Sammle Metriken für jeden Horizont
        horizons = list(evaluation_results.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        # Erstelle Subplots für jede Metrik
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=self.dpi)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Daten für Training und Test
            train_values = []
            test_values = []
            
            for horizon in horizons:
                result = evaluation_results[horizon]
                train_values.append(result['train_metrics'][metric])
                test_values.append(result['test_metrics'][metric])
            
            # Erstelle Balkendiagramm
            x_pos = np.arange(len(horizons))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, train_values, width, label='Training', 
                          color='skyblue', alpha=0.8)
            bars2 = ax.bar(x_pos + width/2, test_values, width, label='Test', 
                          color='lightcoral', alpha=0.8)
            
            # Füge Werte über den Balken hinzu
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Horizont')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} - Training vs Test')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(horizons)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion Matrix Metriken Vergleich gespeichert: {save_path}")
        
        plt.close()
    
    def plot_signal_analysis(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """
        Erstellt Analyse der richtigen und falschen Signale für Training und Test
        
        Args:
            evaluation_results: Evaluationsergebnisse
            save_path: Speicherpfad (optional)
        """
        # Sammle Daten für jeden Horizont
        horizons = list(evaluation_results.keys())
        
        # Erstelle Subplots: 2 Reihen (Training, Test) x Anzahl Horizonte
        fig, axes = plt.subplots(2, len(horizons), figsize=(5*len(horizons), 10), dpi=self.dpi)
        
        # Falls nur ein Horizont, mache axes zu 2D-Array
        if len(horizons) == 1:
            axes = axes.reshape(2, 1)
        
        for i, horizon in enumerate(horizons):
            result = evaluation_results[horizon]
            
            # Training Confusion Matrix
            train_cm = result['train_metrics']['confusion_matrix']
            train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
            
            # Test Confusion Matrix
            test_cm = result['test_metrics']['confusion_matrix']
            test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
            
            # Training Plot
            ax_train = axes[0, i]
            train_correct = [train_tn, train_tp]  # True Negatives, True Positives
            train_incorrect = [train_fp, train_fn]  # False Positives, False Negatives
            
            x = np.arange(2)
            width = 0.35
            
            bars1 = ax_train.bar(x - width/2, train_correct, width, label='Richtige Signale', 
                               color=['lightgreen', 'darkgreen'], alpha=0.8)
            bars2 = ax_train.bar(x + width/2, train_incorrect, width, label='Falsche Signale', 
                               color=['lightcoral', 'darkred'], alpha=0.8)
            
            # Füge Werte über den Balken hinzu
            for bar in bars1:
                height = bar.get_height()
                ax_train.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            for bar in bars2:
                height = bar.get_height()
                ax_train.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax_train.set_xlabel('Signal-Typ')
            ax_train.set_ylabel('Anzahl Signale')
            ax_train.set_title(f'Training - {horizon} Horizont')
            ax_train.set_xticks(x)
            ax_train.set_xticklabels(['Short', 'Long'])
            ax_train.legend()
            ax_train.grid(True, alpha=0.3)
            
            # Test Plot
            ax_test = axes[1, i]
            test_correct = [test_tn, test_tp]  # True Negatives, True Positives
            test_incorrect = [test_fp, test_fn]  # False Positives, False Negatives
            
            bars3 = ax_test.bar(x - width/2, test_correct, width, label='Richtige Signale', 
                              color=['lightgreen', 'darkgreen'], alpha=0.8)
            bars4 = ax_test.bar(x + width/2, test_incorrect, width, label='Falsche Signale', 
                              color=['lightcoral', 'darkred'], alpha=0.8)
            
            # Füge Werte über den Balken hinzu
            for bar in bars3:
                height = bar.get_height()
                ax_test.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            for bar in bars4:
                height = bar.get_height()
                ax_test.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax_test.set_xlabel('Signal-Typ')
            ax_test.set_ylabel('Anzahl Signale')
            ax_test.set_title(f'Test - {horizon} Horizont')
            ax_test.set_xticks(x)
            ax_test.set_xticklabels(['Short', 'Long'])
            ax_test.legend()
            ax_test.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Signal-Analyse gespeichert: {save_path}")
        
        plt.close()
    
    def plot_signal_summary_comparison(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """
        Erstellt Zusammenfassungsvergleich der Signale zwischen Training und Test
        
        Args:
            evaluation_results: Evaluationsergebnisse
            save_path: Speicherpfad (optional)
        """
        # Sammle Daten für jeden Horizont
        horizons = list(evaluation_results.keys())
        
        # Erstelle Subplots für verschiedene Metriken
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        
        # 1. Gesamte richtige vs falsche Signale
        ax1 = axes[0, 0]
        train_correct_total = []
        train_incorrect_total = []
        test_correct_total = []
        test_incorrect_total = []
        
        for horizon in horizons:
            result = evaluation_results[horizon]
            train_cm = result['train_metrics']['confusion_matrix']
            test_cm = result['test_metrics']['confusion_matrix']
            
            train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
            test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
            
            train_correct_total.append(train_tn + train_tp)
            train_incorrect_total.append(train_fp + train_fn)
            test_correct_total.append(test_tn + test_tp)
            test_incorrect_total.append(test_fp + test_fn)
        
        x = np.arange(len(horizons))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, train_correct_total, width, label='Training - Richtige', 
                       color='lightgreen', alpha=0.8)
        bars2 = ax1.bar(x + width/2, test_correct_total, width, label='Test - Richtige', 
                       color='darkgreen', alpha=0.8)
        
        # Füge Werte hinzu
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Horizont')
        ax1.set_ylabel('Anzahl richtige Signale')
        ax1.set_title('Richtige Signale - Training vs Test')
        ax1.set_xticks(x)
        ax1.set_xticklabels(horizons)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Gesamte falsche Signale
        ax2 = axes[0, 1]
        bars3 = ax2.bar(x - width/2, train_incorrect_total, width, label='Training - Falsche', 
                       color='lightcoral', alpha=0.8)
        bars4 = ax2.bar(x + width/2, test_incorrect_total, width, label='Test - Falsche', 
                       color='darkred', alpha=0.8)
        
        # Füge Werte hinzu
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Horizont')
        ax2.set_ylabel('Anzahl falsche Signale')
        ax2.set_title('Falsche Signale - Training vs Test')
        ax2.set_xticks(x)
        ax2.set_xticklabels(horizons)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy-Rate
        ax3 = axes[1, 0]
        train_accuracy = [correct / (correct + incorrect) * 100 
                         for correct, incorrect in zip(train_correct_total, train_incorrect_total)]
        test_accuracy = [correct / (correct + incorrect) * 100 
                        for correct, incorrect in zip(test_correct_total, test_incorrect_total)]
        
        bars5 = ax3.bar(x - width/2, train_accuracy, width, label='Training', 
                       color='skyblue', alpha=0.8)
        bars6 = ax3.bar(x + width/2, test_accuracy, width, label='Test', 
                       color='lightcoral', alpha=0.8)
        
        # Füge Werte hinzu
        for bar in bars5:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar in bars6:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.set_xlabel('Horizont')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Accuracy-Rate - Training vs Test')
        ax3.set_xticks(x)
        ax3.set_xticklabels(horizons)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. Signal-Verhältnis (Richtige/Falsche)
        ax4 = axes[1, 1]
        train_ratio = [correct / incorrect if incorrect > 0 else 0 
                      for correct, incorrect in zip(train_correct_total, train_incorrect_total)]
        test_ratio = [correct / incorrect if incorrect > 0 else 0 
                     for correct, incorrect in zip(test_correct_total, test_incorrect_total)]
        
        bars7 = ax4.bar(x - width/2, train_ratio, width, label='Training', 
                       color='lightgreen', alpha=0.8)
        bars8 = ax4.bar(x + width/2, test_ratio, width, label='Test', 
                       color='darkgreen', alpha=0.8)
        
        # Füge Werte hinzu
        for bar in bars7:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar in bars8:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax4.set_xlabel('Horizont')
        ax4.set_ylabel('Verhältnis (Richtige/Falsche)')
        ax4.set_title('Signal-Qualitätsverhältnis - Training vs Test')
        ax4.set_xticks(x)
        ax4.set_xticklabels(horizons)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Signal-Zusammenfassungsvergleich gespeichert: {save_path}")
        
        plt.close()
    
    def plot_heatmap_accuracy(self, results_summary: pd.DataFrame, save_path: Optional[str] = None):
        """
        Erstellt Heatmap der Modellgüte über verschiedene Zeithorizonte
        
        Args:
            results_summary: Zusammenfassung der Ergebnisse
            save_path: Speicherpfad (optional)
        """
        # Pivot-Tabelle für Heatmap
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        heatmap_data = results_summary.set_index('horizon')[metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, ax=ax)
        
        ax.set_title('Modellgüte über verschiedene Zeithorizonte')
        ax.set_xlabel('Prognosehorizont')
        ax.set_ylabel('Metrik')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Accuracy Heatmap gespeichert: {save_path}")
        
        plt.close()
    
    def create_interactive_dashboard(self, results_df: pd.DataFrame, 
                                   evaluation_results: Dict) -> go.Figure:
        """
        Erstellt ein interaktives Dashboard mit Plotly
        
        Args:
            results_df: Ergebnisse DataFrame
            evaluation_results: Evaluationsergebnisse
            
        Returns:
            Plotly Figure mit Dashboard
        """
        # Erstelle Subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Wahrscheinlichkeiten über Zeit', 'Kursänderungen über Zeit',
                          'ROC-Kurven', 'Accuracy Heatmap',
                          'Intensitätsverteilung', 'Feature Importance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Wahrscheinlichkeiten
        for horizon in Config.FORECAST_HORIZONS.keys():
            prob_col = f'{horizon}_probability'
            if prob_col in results_df.columns:
                fig.add_trace(
                    go.Scatter(x=results_df['publication_date'], 
                              y=results_df[prob_col],
                              mode='lines', name=f'{horizon} Wahrscheinlichkeit',
                              opacity=0.7),
                    row=1, col=1
                )
        
        # Plot 2: Kursänderungen
        for horizon in Config.FORECAST_HORIZONS.keys():
            rate_change_col = f'{horizon}_rate_change'
            if rate_change_col in results_df.columns:
                fig.add_trace(
                    go.Scatter(x=results_df['publication_date'], 
                              y=results_df[rate_change_col],
                              mode='lines', name=f'{horizon} Kursänderung',
                              opacity=0.7),
                    row=1, col=2
                )
        
        # Plot 3: ROC-Kurven
        for horizon, result in evaluation_results.items():
            if 'roc_curve' in result['test_metrics']:
                roc_data = result['test_metrics']['roc_curve']
                auc = result['test_metrics']['roc_auc']
                
                fig.add_trace(
                    go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'],
                              mode='lines', name=f'{horizon} (AUC={auc:.3f})'),
                    row=2, col=1
                )
        
        # Plot 4: Accuracy Heatmap
        results_summary = pd.DataFrame([
            {
                'horizon': horizon,
                'accuracy': result['test_metrics']['accuracy'],
                'precision': result['test_metrics']['precision'],
                'recall': result['test_metrics']['recall'],
                'f1_score': result['test_metrics']['f1_score']
            }
            for horizon, result in evaluation_results.items()
        ])
        
        if not results_summary.empty:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            heatmap_data = results_summary.set_index('horizon')[metrics].values
            
            fig.add_trace(
                go.Heatmap(z=heatmap_data, x=metrics, 
                          y=results_summary['horizon'],
                          colorscale='RdYlGn'),
                row=2, col=2
            )
        
        # Plot 5: Intensitätsverteilung
        for horizon in Config.FORECAST_HORIZONS.keys():
            intensity_col = f'{horizon}_intensity'
            if intensity_col in results_df.columns:
                intensity_counts = results_df[intensity_col].value_counts()
                
                fig.add_trace(
                    go.Bar(x=intensity_counts.index, y=intensity_counts.values,
                           name=f'{horizon} Intensität'),
                    row=3, col=1
                )
        
        # Layout anpassen
        fig.update_layout(
            title_text="Devisenhandel-Vorhersagesystem Dashboard",
            height=1200,
            showlegend=True
        )
        
        return fig

class Validator:
    """Klasse zur Validierung der Ergebnisse"""
    
    def __init__(self):
        """Initialisiert den Validator"""
        self.validation_results = {}
    
    def validate_model_performance(self, evaluation_results: Dict) -> Dict:
        """
        Validiert die Modellleistung
        
        Args:
            evaluation_results: Evaluationsergebnisse
            
        Returns:
            Dictionary mit Validierungsergebnissen
        """
        logger.info("Starte Modellvalidierung")
        
        validation_results = {}
        
        for horizon, result in evaluation_results.items():
            test_metrics = result['test_metrics']
            
            # Basis-Validierung
            validation = {
                'horizon': horizon,
                'accuracy_above_random': test_metrics['accuracy'] > 0.5,
                'auc_above_random': test_metrics['roc_auc'] > 0.5,
                'precision_acceptable': test_metrics['precision'] > 0.4,
                'recall_acceptable': test_metrics['recall'] > 0.4,
                'f1_acceptable': test_metrics['f1_score'] > 0.4
            }
            
            # Erweiterte Validierung
            validation['overall_performance'] = (
                validation['accuracy_above_random'] and 
                validation['auc_above_random'] and
                validation['precision_acceptable'] and
                validation['recall_acceptable']
            )
            
            validation_results[horizon] = validation
        
        self.validation_results = validation_results
        logger.info("Modellvalidierung abgeschlossen")
        return validation_results
    
    def perform_statistical_tests(self, evaluation_results: Dict) -> Dict:
        """
        Führt statistische Tests zwischen den Zeithorizonten durch
        
        Args:
            evaluation_results: Evaluationsergebnisse
            
        Returns:
            Dictionary mit Testergebnissen
        """
        logger.info("Führe statistische Tests durch")
        
        # McNemar-Test für binäre Klassifikation
        # (Vereinfachte Implementierung - in der Praxis würde man die tatsächlichen Vorhersagen verwenden)
        
        test_results = {
            'horizon_comparison': {},
            'performance_ranking': {}
        }
        
        # Performance-Ranking
        performance_scores = {}
        for horizon, result in evaluation_results.items():
            test_metrics = result['test_metrics']
            # Gewichteter Score
            score = (test_metrics['accuracy'] * 0.3 + 
                    test_metrics['precision'] * 0.2 + 
                    test_metrics['recall'] * 0.2 + 
                    test_metrics['f1_score'] * 0.2 + 
                    test_metrics['roc_auc'] * 0.1)
            performance_scores[horizon] = score
        
        # Sortiere nach Performance
        sorted_horizons = sorted(performance_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        test_results['performance_ranking'] = dict(sorted_horizons)
        
        logger.info("Statistische Tests abgeschlossen")
        return test_results

class OutputModule:
    """Hauptklasse des Output- und Evaluationsmoduls"""
    
    def __init__(self):
        """Initialisiert das Output-Modul"""
        self.processor = ResultProcessor()
        self.visualizer = Visualizer()
        self.validator = Validator()
    
    def _create_results_summary_from_evaluation_results(self, evaluation_results: Dict) -> pd.DataFrame:
        """
        Erstellt eine Zusammenfassung der Ergebnisse aus evaluation_results
        Args:
            evaluation_results: Dictionary mit Evaluationsergebnissen
        Returns:
            DataFrame mit Ergebnisübersicht
        """
        summary_data = []

        for horizon, result in evaluation_results.items():
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
    
    def _create_comparison_summary_from_evaluation_results(self, evaluation_results: Dict) -> pd.DataFrame:
        """
        Erstellt eine Vergleichszusammenfassung aus evaluation_results
        Args:
            evaluation_results: Dictionary mit Evaluationsergebnissen
        Returns:
            DataFrame mit Vergleichsübersicht
        """
        comparison_data = []

        for horizon, result in evaluation_results.items():
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
    
    def _convert_numpy_types(self, obj):
        """
        Konvertiert NumPy-Datentypen zu Python-native Typen für JSON-Serialisierung
        """
        import numpy as np
        
        try:
            if isinstance(obj, dict):
                return {key: self._convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [self._convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, (pd.DatetimeTZDtype, pd.Timestamp)):
                return str(obj)
            elif hasattr(obj, 'dtype') and str(obj.dtype).startswith('datetime'):
                return str(obj)
            elif hasattr(obj, 'dtype') and str(obj.dtype).startswith('timedelta'):
                return str(obj)
            elif str(type(obj)).startswith("<class 'pandas."):
                return str(obj)
            elif str(type(obj)).startswith("<class 'numpy."):
                return str(obj)
            else:
                return obj
        except Exception:
            return str(obj)
        
    def process_and_analyze(self, predictions_df: pd.DataFrame,
                          event_matrix: pd.DataFrame,
                          evaluation_results: Dict) -> Dict:
        """
        Vollständige Ergebnisverarbeitung und -analyse
        
        Args:
            predictions_df: Vorhersagen
            event_matrix: Ereignismatrix
            evaluation_results: Evaluationsergebnisse
            
        Returns:
            Dictionary mit allen Ergebnissen
        """
        logger.info("Starte vollständige Ergebnisanalyse")
        
        # 1. Ergebnisverarbeitung
        results_df = self.processor.process_predictions(predictions_df, event_matrix)
        
        # 2. Umfassende Variablen-Zusammenfassung
        comprehensive_variable_summary = self.processor.create_comprehensive_variable_summary(event_matrix)
        
        # 3. Validierung
        validation_results = self.validator.validate_model_performance(evaluation_results)
        statistical_tests = self.validator.perform_statistical_tests(evaluation_results)
        
        # 3. Zusammenfassung erstellen (mit Trainings- und Test-Ergebnissen)
        # Erstelle Zusammenfassungen aus den evaluation_results
        results_summary = self._create_results_summary_from_evaluation_results(evaluation_results)
        comparison_summary = self._create_comparison_summary_from_evaluation_results(evaluation_results)
        
        # Füge Validierungsergebnisse hinzu
        validation_info = []
        for horizon, result in evaluation_results.items():
            if horizon in validation_results:
                validation_info.append({
                    'horizon': horizon,
                    'validation_passed': validation_results[horizon]['overall_performance']
                })
        
        validation_df = pd.DataFrame(validation_info)
        if not validation_df.empty:
            results_summary = results_summary.merge(validation_df, on='horizon', how='left')
        
        # 4. Ergebnisse zusammenfassen
        complete_results = {
            'processed_results': results_df,
            'evaluation_results': evaluation_results,
            'validation_results': validation_results,
            'statistical_tests': statistical_tests,
            'results_summary': results_summary,
            'comparison_summary': comparison_summary,
            'summary_statistics': self.processor.results.get('summary_statistics', {}),
            'comprehensive_variable_summary': comprehensive_variable_summary
        }
        
        logger.info("Ergebnisanalyse abgeschlossen")
        return complete_results
    
    def create_visualizations(self, results_df: pd.DataFrame, 
                            evaluation_results: Dict,
                            save_plots: bool = True) -> Dict:
        """
        Erstellt alle Visualisierungen
        
        Args:
            results_df: Verarbeitete Ergebnisse
            evaluation_results: Evaluationsergebnisse
            save_plots: Ob Plots gespeichert werden sollen
            
        Returns:
            Dictionary mit Plot-Pfaden
        """
        logger.info("Erstelle Visualisierungen")
        
        plot_paths = {}
        
        # Zeitreihenplots für jeden Horizont
        for horizon in Config.FORECAST_HORIZONS.keys():
            if save_plots:
                save_path = f"{Config.PLOTS_DIR}/time_series_{horizon}.png"
                self.visualizer.plot_time_series(results_df, horizon, save_path)
                plot_paths[f'time_series_{horizon}'] = save_path
            else:
                self.visualizer.plot_time_series(results_df, horizon)
        
        # ROC-Kurven
        if save_plots:
            save_path = f"{Config.PLOTS_DIR}/roc_curves.png"
            self.visualizer.plot_roc_curves(evaluation_results, save_path)
            plot_paths['roc_curves'] = save_path
        else:
            self.visualizer.plot_roc_curves(evaluation_results)
        
        # Confusion Matrizen
        if save_plots:
            save_path = f"{Config.PLOTS_DIR}/confusion_matrices.png"
            self.visualizer.plot_confusion_matrices(evaluation_results, save_path)
            plot_paths['confusion_matrices'] = save_path
        else:
            self.visualizer.plot_confusion_matrices(evaluation_results)
        
        # Accuracy Heatmap
        results_summary = pd.DataFrame([
            {
                'horizon': horizon,
                'accuracy': result['test_metrics']['accuracy'],
                'precision': result['test_metrics']['precision'],
                'recall': result['test_metrics']['recall'],
                'f1_score': result['test_metrics']['f1_score'],
                'roc_auc': result['test_metrics']['roc_auc']
            }
            for horizon, result in evaluation_results.items()
        ])
        
        if save_plots and not results_summary.empty:
            save_path = f"{Config.PLOTS_DIR}/accuracy_heatmap.png"
            self.visualizer.plot_heatmap_accuracy(results_summary, save_path)
            plot_paths['accuracy_heatmap'] = save_path
        elif not results_summary.empty:
            self.visualizer.plot_heatmap_accuracy(results_summary)
        
        # Interaktives Dashboard
        dashboard = self.visualizer.create_interactive_dashboard(results_df, evaluation_results)
        if save_plots:
            save_path = f"{Config.PLOTS_DIR}/interactive_dashboard.html"
            dashboard.write_html(save_path)
            plot_paths['interactive_dashboard'] = save_path
        
        logger.info("Visualisierungen erstellt")
        return plot_paths
    
    def save_results(self, complete_results: Dict, filename_prefix: str):
        """
        Speichert alle Ergebnisse
        
        Args:
            complete_results: Vollständige Ergebnisse
            filename_prefix: Präfix für Dateinamen
        """
        logger.info("Speichere Ergebnisse")
        
        # Verarbeitete Ergebnisse
        results_path = f"{Config.RESULTS_DIR}/{filename_prefix}_processed_results.csv"
        complete_results['processed_results'].to_csv(results_path, index=False)
        
        # Zusammenfassung
        summary_path = f"{Config.RESULTS_DIR}/{filename_prefix}_results_summary.csv"
        complete_results['results_summary'].to_csv(summary_path, index=False)
        
        # Feature Importance für jeden Horizont
        for horizon, result in complete_results['evaluation_results'].items():
            importance_path = f"{Config.RESULTS_DIR}/{filename_prefix}_{horizon}_feature_importance.csv"
            result['feature_importance'].to_csv(importance_path, index=False)
            
            # Regressionsergebnisse-Tabelle speichern
            if 'regression_table' in result and not result['regression_table'].empty:
                regression_path = f"{Config.RESULTS_DIR}/{filename_prefix}_{horizon}_regression_results.csv"
                result['regression_table'].to_csv(regression_path, index=False)
                logger.info(f"Regressionsergebnisse für {horizon} gespeichert: {regression_path}")
        
        # Train-only Feature Importance (alle Daten aus dem Plot)
        train_rows = []
        for horizon, result in complete_results['evaluation_results'].items():
            train_importance = result.get('train_feature_importance', pd.DataFrame())
            if train_importance is None or train_importance.empty:
                continue
            # Filtern wie im Plot: 'indicator_value_standardized' ausblenden
            train_importance = train_importance[train_importance['feature'] != 'indicator_value_standardized']
            for _, row in train_importance.iterrows():
                feature_name = row['feature']
                translated = self.visualizer.translate_feature_name(feature_name) if hasattr(self, 'visualizer') else feature_name
                train_rows.append({
                    'horizon': horizon,
                    'feature': feature_name,
                    'feature_translated': translated,
                    'coefficient': row.get('coefficient'),
                    'abs_coefficient': row.get('abs_coefficient'),
                    'std_error': row.get('std_error'),
                    'z_value': row.get('z_value'),
                    'p_value': row.get('p_value'),
                    'significance': row.get('significance')
                })
        if train_rows:
            train_df = pd.DataFrame(train_rows)
            train_all_path = f"{Config.RESULTS_DIR}/{filename_prefix}_train_feature_importance_all.csv"
            train_df.to_csv(train_all_path, index=False)
            logger.info(f"Train Feature-Importance (alle Daten) gespeichert: {train_all_path}")

        # Test-only Feature Importance (CSV)
        test_rows = []
        for horizon, result in complete_results['evaluation_results'].items():
            test_importance = result.get('test_feature_importance', pd.DataFrame())
            if test_importance is None or test_importance.empty:
                continue
            test_importance = test_importance[test_importance['feature'] != 'indicator_value_standardized']
            for _, row in test_importance.iterrows():
                feature_name = row['feature']
                translated = self.visualizer.translate_feature_name(feature_name) if hasattr(self, 'visualizer') else feature_name
                test_rows.append({
                    'horizon': horizon,
                    'feature': feature_name,
                    'feature_translated': translated,
                    'coefficient': row.get('coefficient'),
                    'abs_coefficient': row.get('abs_coefficient'),
                    'std_error': row.get('std_error'),
                    'z_value': row.get('z_value'),
                    'p_value': row.get('p_value'),
                    'significance': row.get('significance')
                })
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            test_all_path = f"{Config.RESULTS_DIR}/{filename_prefix}_test_feature_importance_all.csv"
            test_df.to_csv(test_all_path, index=False)
            logger.info(f"Test Feature-Importance (alle Daten) gespeichert: {test_all_path}")

        # Validierungsergebnisse
        validation_path = f"{Config.RESULTS_DIR}/{filename_prefix}_validation_results.csv"
        validation_df = pd.DataFrame([
            {
                'horizon': horizon,
                **validation_data
            }
            for horizon, validation_data in complete_results['validation_results'].items()
        ])
        validation_df.to_csv(validation_path, index=False)
        
        # Vergleichszusammenfassung speichern
        comparison_path = f"{Config.RESULTS_DIR}/{filename_prefix}_comparison_summary.csv"
        complete_results['comparison_summary'].to_csv(comparison_path, index=False)
        logger.info(f"Vergleichszusammenfassung gespeichert: {comparison_path}")
        
        # Umfassende Variablen-Zusammenfassung
        variable_summary_path = f"{Config.RESULTS_DIR}/{filename_prefix}_comprehensive_variable_summary.json"
        with open(variable_summary_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_numpy_types(complete_results['comprehensive_variable_summary']), f, indent=4)
        logger.info(f"Umfassende Variablen-Zusammenfassung gespeichert: {variable_summary_path}")
        
        # Erstelle neue Visualisierungen für Training vs Test Vergleich
        logger.info("Erstelle neue Visualisierungen für Training vs Test Vergleich")
        
        # 1. Training vs Test Vergleich
        training_test_plot_path = f"{Config.PLOTS_DIR}/{filename_prefix}_training_test_comparison.png"
        self.visualizer.plot_training_test_comparison(
            complete_results['results_summary'], 
            save_path=training_test_plot_path
        )
        
        # 2. Overfitting-Analyse
        overfitting_plot_path = f"{Config.PLOTS_DIR}/{filename_prefix}_overfitting_analysis.png"
        self.visualizer.plot_overfitting_analysis(
            complete_results['comparison_summary'], 
            save_path=overfitting_plot_path
        )
        
        # 3. Metriken Heatmap
        metrics_heatmap_path = f"{Config.PLOTS_DIR}/{filename_prefix}_metrics_heatmap.png"
        self.visualizer.plot_metrics_heatmap(
            complete_results['results_summary'], 
            save_path=metrics_heatmap_path
        )
        
        # 4. Feature-Importance Vergleich
        feature_importance_comparison_path = f"{Config.PLOTS_DIR}/{filename_prefix}_feature_importance_comparison.png"
        self.visualizer.plot_feature_importance_comparison(
            complete_results['evaluation_results'], 
            save_path=feature_importance_comparison_path
        )
        
        # 5. Feature-Importance Heatmap
        feature_importance_heatmap_path = f"{Config.PLOTS_DIR}/{filename_prefix}_feature_importance_heatmap.png"
        self.visualizer.plot_feature_importance_heatmap(
            complete_results['evaluation_results'], 
            save_path=feature_importance_heatmap_path
        )

        # 6. Nur Trainings-Feature-Importance
        train_only_feature_importance_path = f"{Config.PLOTS_DIR}/{filename_prefix}_train_feature_importance.png"
        self.visualizer.plot_feature_importance_training_only(
            complete_results['evaluation_results'],
            save_path=train_only_feature_importance_path
        )

        # 7. Nur Test-Feature-Importance
        test_only_feature_importance_path = f"{Config.PLOTS_DIR}/{filename_prefix}_test_feature_importance.png"
        self.visualizer.plot_feature_importance_test_only(
            complete_results['evaluation_results'],
            save_path=test_only_feature_importance_path
        )
        
        # 6. Training vs Test Confusion Matrizen
        confusion_matrices_path = f"{Config.PLOTS_DIR}/{filename_prefix}_confusion_matrices.png"
        self.visualizer.plot_confusion_matrices(
            complete_results['evaluation_results'], 
            save_path=confusion_matrices_path
        )
        
        # 7. Confusion Matrix Metriken Vergleich
        confusion_metrics_comparison_path = f"{Config.PLOTS_DIR}/{filename_prefix}_confusion_metrics_comparison.png"
        self.visualizer.plot_confusion_matrix_metrics_comparison(
            complete_results['evaluation_results'], 
            save_path=confusion_metrics_comparison_path
        )
        
        # 8. Signal-Analyse (Detaillierte Analyse der richtigen/falschen Signale)
        signal_analysis_path = f"{Config.PLOTS_DIR}/{filename_prefix}_signal_analysis.png"
        self.visualizer.plot_signal_analysis(
            complete_results['evaluation_results'], 
            save_path=signal_analysis_path
        )
        
        # 9. Signal-Zusammenfassungsvergleich
        signal_summary_comparison_path = f"{Config.PLOTS_DIR}/{filename_prefix}_signal_summary_comparison.png"
        self.visualizer.plot_signal_summary_comparison(
            complete_results['evaluation_results'], 
            save_path=signal_summary_comparison_path
        )
        
        logger.info("Neue Visualisierungen erstellt und gespeichert")
        logger.info("Ergebnisse gespeichert")
    
    def generate_report(self, complete_results: Dict, filename_prefix: str) -> str:
        """
        Generiert einen detaillierten Bericht
        
        Args:
            complete_results: Vollständige Ergebnisse
            filename_prefix: Präfix für Dateinamen
            
        Returns:
            Bericht als String
        """
        logger.info("Generiere Bericht")
        
        report = []
        report.append("=" * 80)
        report.append("DEVISENHANDEL-VORHERGESYSTEM - ERGEBNISBERICHT")
        report.append("=" * 80)
        report.append(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("METHODISCHE GRUNDLAGE")
        report.append("-" * 40)
        report.append("• Logistische Regression für binäre Klassifikation")
        report.append("• Schwellenwert: 0.25% für signifikante Kursbewegungen")
        report.append("• Training: 2003-2019, Test: 2020-2025")
        report.append("• Prognosehorizonte: 1, 5, 20, 250 Handelstage")
        report.append("• Standardisierung: Z-Transformation")
        report.append("")

        # As-of-Konstruktion und Look-Ahead-Vermeidung
        report.append("AS-OF-KONSTRUKTION DER PRÄDIKTOREN UND LOOK-AHEAD-VERMEIDUNG")
        report.append("-" * 40)
        report.append("Alle Prädiktoren werden strikt as-of zum Prognosezeitpunkt t gebildet.")
        report.append("Für jeden Handelstag t gilt: jede Variable erhält den zuletzt bis t veröffentlichten Wert.")
        report.append("Zwischen Veröffentlichungen erfolgt keine Interpolation, sondern eine konstante Fortschreibung (LOCF).")
        report.append("")
        report.append("Formell: Sei {τ_j} die Menge der Veröffentlichungszeitpunkte eines Indikators X.")
        report.append("Dann gilt:  X_t := X_{τ_j}  mit  τ_j = max{ τ ∈ {Veröffentlichungszeiten} | τ ≤ t }.")
        report.append("Damit ist X_t F_t-messbar und enthält keine zukünftige Information (kein Look-Ahead).")
        report.append("")
        report.append("Veröffentlichungsfrequenz vs. Prognosehorizont")
        report.append("Der Zeitindex t ist der Handelstag. Frequenz (monatlich/vierteljährlich) und Horizont (Tage/Wochen/Monate/Jahre) sind unabhängig,")
        report.append("da X_t ausschließlich auf {τ ≤ t} beruht und somit keine Zukunftsinformation einspeist.")
        report.append("")
        report.append("Umgang mit unterschiedlichen Frequenzen")
        report.append("  • Quartalsdaten (z. B. BIP): ab Veröffentlichung τ bis zur nächsten Veröffentlichung konstant (Stufenfunktion).")
        report.append("  • Monatsdaten (z. B. CPI, Arbeitslosenquote): analog; keine lineare/Spline-Interpolation auf tägliche Werte.")
        report.append("  • Wöchentliche/tägliche Daten (z. B. Zinsen): tagesgenau; Aggregationen (z. B. 5-Tage-Schnitt) nutzen nur Historie ≤ t.")
        report.append("")

        # 3.1 Datenauswahl und Quellen
        report.append("3.1 DATENAUSWAHL UND QUELLEN")
        report.append("-" * 40)
        report.append("Die Auswahl der ökonomischen Indikatoren erfolgt auf Basis ihrer theoretischen Relevanz ")
        report.append("und praktischen Bedeutung für die Wechselkursbildung im Devisenhandel.")
        report.append("Datenquellen:")
        report.append("  • FRED (Federal Reserve Bank of St. Louis) via fredapi für Makrodaten")
        report.append("  • yfinance für historische EUR/USD‑Wechselkursdaten")
        report.append("Diese Kombination erlaubt es, tagesgenaue Kursbewegungen systematisch mit ")
        report.append("den Veröffentlichungszeitpunkten makroökonomischer Kennzahlen zu verknüpfen.")
        report.append("")

        # 3.2 Modellstruktur und Zielvariable
        report.append("3.2 MODELLSTRUKTUR UND ZIELVARIABLE")
        report.append("-" * 40)
        report.append("Getestete Prognosehorizonte h:")
        report.append("  • Tagesmodell: h = 1 Handelstag")
        report.append("  • Wochenmodell: h = 5 Handelstage")
        report.append("  • Monatsmodell: h = 20 Handelstage")
        report.append("  • Jahresmodell: h = 250 Handelstage")
        report.append("Die unabhängigen Variablen x_t enthalten alle zu t verfügbaren Werte der Indikatoren.")
        report.append("Die Zielvariable {Y_t}^h klassifiziert die Kursbewegung über den Horizont h binär.")
        report.append("Klassifikationsregel (mit Schwellenwert θ = ±0,6378%):")
        report.append("  • Y_t^h = 1 (Long‑Bias), wenn Δ_{t,t+h} ≥ +0,6378%")
        report.append("  • Y_t^h = 0 (Short‑Bias), wenn Δ_{t,t+h} ≤ −0,6378%")
        report.append("  • Beobachtungen mit |Δ_{t,t+h}| < 0,6378% werden neutral behandelt und ausgeschlossen.")
        report.append("Definition der relativen Kursänderung (in %):  Δ_{t,t+h} = (S_{t+h} − S_t)/S_t * 100")
        report.append("")
        report.append("Begründung des Schwellenwerts")
        report.append("Der Schwellenwert ±0,6378% orientiert sich am Median der Intraday‑Range ")
        report.append("(Tageshoch−Tagestief relativ zum Eröffnungskurs) im Zeitraum 2003–2025.")
        report.append("Damit werden typische Rauschbewegungen gefiltert und die Klassifikation auf ")
        report.append("ökonomisch relevante Impulse fokussiert (Directional‑Change‑Literatur; ")
        report.append("volatilitätsbasierte Jump‑Tests; praxisübliche Filterregeln im FX‑Handel).")
        report.append("")

        # 3.3 Logistische Regression
        report.append("3.3 LOGISTISCHE REGRESSION ALS PROGNOSEMODELL")
        report.append("-" * 40)
        report.append("Das Modell schätzt die Wahrscheinlichkeit eines Long‑Bias:")
        report.append("  P(Y_t^h = 1) = 1 / (1 + exp(−(β_0 + β_1 x_{1,t} + … + β_n x_{n,t})))")
        report.append("Die Koeffizienten β_i geben Richtung und Stärke des Einflusses an.")
        report.append("Schätzung auf Training (2003–2019), Evaluierung auf Test (2020–2025).")
        report.append("")
        report.append("WICHTIGE METHODISCHE ÄNDERUNG:")
        report.append("Die unabhängigen Variablen werden als Log-Differenzen (log(1 + x)) transformiert,")
        report.append("um die Interpretation der Koeffizienten zu verbessern und nicht-lineare")
        report.append("Zusammenhänge zu erfassen. Dies entspricht dem Standard in der Finance-Literatur.")
        report.append("")
        
        # Regressionsergebnisse-Tabellen
        report.append("REGRESSIONSERGEBNISSE")
        report.append("-" * 40)
        report.append("Die folgenden Tabellen zeigen die Koeffizienten, Standardfehler, t-Statistiken")
        report.append("und Signifikanzlevels für jeden Prognosehorizont. Signifikanzlevels:")
        report.append("*** p<0.001, ** p<0.01, * p<0.05, • p<0.1")
        report.append("")
        
        for horizon in Config.FORECAST_HORIZONS.keys():
            if horizon in complete_results['evaluation_results']:
                result = complete_results['evaluation_results'][horizon]
                if 'regression_table' in result and not result['regression_table'].empty:
                    regression_table = result['regression_table']
                    
                    report.append(f"TABELLE: Regressionsergebnisse für {horizon.upper()}-Horizont")
                    report.append("-" * 60)
                    report.append(f"{'Variable':<25} {'Coeff':<8} {'StdErr':<8} {'t-Stat':<8} {'P-Value':<8} {'Sig':<4}")
                    report.append("-" * 60)
                    
                    for _, row in regression_table.head(10).iterrows():  # Top 10 Variablen
                        var_name = row['Variable'][:24]  # Kürze für bessere Darstellung
                        coeff = f"{row['Coefficient']:.4f}"
                        stderr = f"{row['Std_Error']:.4f}"
                        tstat = f"{row['t_Statistic']:.3f}"
                        pval = f"{row['P_Value']:.4f}"
                        sig = row['Significance']
                        
                        report.append(f"{var_name:<25} {coeff:<8} {stderr:<8} {tstat:<8} {pval:<8} {sig:<4}")
                    
                    if len(regression_table) > 10:
                        report.append(f"... und {len(regression_table) - 10} weitere Variablen")
                    
                    report.append("")
                    report.append(f"Anzahl Beobachtungen: {result.get('n_train_samples', 'N/A')}")
                    report.append(f"Anzahl Variablen: {result.get('n_features', 'N/A')}")
                    report.append("")
        
        # Zusammenfassung (Training vs Test)
        report.append("ZUSAMMENFASSUNG - TRAINING VS TEST")
        report.append("-" * 40)
        results_summary = complete_results['results_summary']
        comparison_summary = complete_results['comparison_summary']
        
        # Gruppiere nach Horizont
        for horizon in Config.FORECAST_HORIZONS.keys():
            train_data = results_summary[(results_summary['horizon'] == horizon) & 
                                       (results_summary['sample_type'] == 'Training')]
            test_data = results_summary[(results_summary['horizon'] == horizon) & 
                                      (results_summary['sample_type'] == 'Test')]
            
            if not train_data.empty and not test_data.empty:
                train_row = train_data.iloc[0]
                test_row = test_data.iloc[0]
                
                report.append(f"Horizont: {horizon}")
                report.append(f"  Trainingsdaten: {train_row['start_date']} bis {train_row['end_date']} ({train_row['n_samples']} Samples)")
                report.append(f"  Testdaten: {test_row['start_date']} bis {test_row['end_date']} ({test_row['n_samples']} Samples)")
                report.append("")
                report.append("  TRAINING:")
                report.append(f"    Accuracy: {train_row['accuracy']:.4f}")
                report.append(f"    Precision: {train_row['precision']:.4f}")
                report.append(f"    Recall: {train_row['recall']:.4f}")
                report.append(f"    F1-Score: {train_row['f1_score']:.4f}")
                report.append(f"    ROC-AUC: {train_row['roc_auc']:.4f}")
                report.append("")
                report.append("  TEST:")
                report.append(f"    Accuracy: {test_row['accuracy']:.4f}")
                report.append(f"    Precision: {test_row['precision']:.4f}")
                report.append(f"    Recall: {test_row['recall']:.4f}")
                report.append(f"    F1-Score: {test_row['f1_score']:.4f}")
                report.append(f"    ROC-AUC: {test_row['roc_auc']:.4f}")
                
                # Overfitting-Analyse
                comp_row = comparison_summary[comparison_summary['horizon'] == horizon]
                if not comp_row.empty:
                    comp = comp_row.iloc[0]
                    report.append("")
                    report.append("  VERGLEICH:")
                    report.append(f"    Accuracy-Differenz (Test-Training): {comp['accuracy_diff']:.4f}")
                    report.append(f"    Precision-Differenz (Test-Training): {comp['precision_diff']:.4f}")
                    report.append(f"    Recall-Differenz (Test-Training): {comp['recall_diff']:.4f}")
                    report.append(f"    F1-Differenz (Test-Training): {comp['f1_diff']:.4f}")
                    report.append(f"    ROC-AUC-Differenz (Test-Training): {comp['roc_auc_diff']:.4f}")
                    report.append(f"    Overfitting-Indikator: {comp['overfitting_indicator']}")
                
                report.append("")
                report.append("-" * 40)
                report.append("")
        
        # Performance-Ranking
        report.append("PERFORMANCE-RANKING")
        report.append("-" * 40)
        ranking = complete_results['statistical_tests']['performance_ranking']
        for i, (horizon, score) in enumerate(ranking.items(), 1):
            report.append(f"{i}. {horizon}: {score:.4f}")
        report.append("")
        
        # Validierungsergebnisse
        report.append("VALIDIERUNGSERGEBNISSE")
        report.append("-" * 40)
        validation_results = complete_results['validation_results']
        for horizon, validation in validation_results.items():
            report.append(f"Horizont: {horizon}")
            report.append(f"  Accuracy über Zufall: {validation['accuracy_above_random']}")
            report.append(f"  AUC über Zufall: {validation['auc_above_random']}")
            report.append(f"  Precision akzeptabel: {validation['precision_acceptable']}")
            report.append(f"  Recall akzeptabel: {validation['recall_acceptable']}")
            report.append(f"  Gesamtleistung: {validation['overall_performance']}")
            report.append("")
        
        # Empfehlungen
        report.append("EMPFOHLUNGEN")
        report.append("-" * 40)
        
        best_horizon = max(ranking.items(), key=lambda x: x[1])[0]
        report.append(f"Bester Horizont: {best_horizon}")
        
        passed_validation = [h for h, v in validation_results.items() 
                           if v['overall_performance']]
        if passed_validation:
            report.append(f"Empfohlene Horizonte für Trading: {', '.join(passed_validation)}")
        else:
            report.append("Kein Horizont erfüllt alle Validierungskriterien")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Speichere Bericht
        report_path = f"{Config.RESULTS_DIR}/{filename_prefix}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Bericht gespeichert: {report_path}")
        return report_text 