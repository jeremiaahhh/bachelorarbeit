"""
Umfassende statistische Analyse aller verwendeten Variablen
Erstellt detaillierte Statistiken für Zinsspread, Inflation, Arbeitsmarkt, BIP usw.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List
import json
import os

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, some statistical tests will be skipped")

from input_module import InputModule
from config import Config

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VariableStatisticsAnalyzer:
    """Klasse zur statistischen Analyse aller Variablen"""
    
    def __init__(self):
        """Initialisiert den Analyzer"""
        self.input_module = InputModule()
        
        # Deutsche Bezeichnungen für Variablen
        self.variable_names = {
            # Basis-Indikatoren
            'GDP': 'BIP USA',
            'CLVMNACSCAB1GQEU28': 'BIP Eurozone',
            'CPIAUCSL': 'Verbraucherpreisindex USA',
            'CP0000EZ19M086NEST': 'Verbraucherpreisindex Eurozone',
            'PPIACO': 'Produzentenpreisindex USA',
            'PIEAMP01EUM661N': 'Produzentenpreisindex Eurozone',
            'FEDFUNDS': 'Fed Funds Rate',
            'IR3TIB01EZM156N': '3-Monats EURIBOR',
            'UNRATE': 'Arbeitslosenquote USA',
            'LRUNTTTTDEQ156S': 'Arbeitslosenquote Deutschland',
            'UMCSENT': 'Verbrauchervertrauen USA',
            'BSCICP03EZM665S': 'Verbrauchervertrauen Eurozone',
            
            # Abgeleitete Indikatoren
            'UNRATE_diff': 'US-Arbeitslosenquote (Monatsdifferenz)',
            'LRUNTTTTDEQ156S_diff': 'DE-Arbeitslosenquote (Monatsdifferenz)',
            'INTEREST_RATE_SPREAD': 'Zinsspread (USA-EU)',
            'GDP_growth': 'US-BIP-Wachstum (Jahresrate)',
            'CLVMNACSCAB1GQEU28_growth': 'EU-BIP-Wachstum (Jahresrate)',
            'CPIAUCSL_yoy': 'US-Inflation (Jahresveränderung)',
            'CP0000EZ19M086NEST_yoy': 'EU-Inflation (Jahresveränderung)',
            'PPIACO_yoy': 'US-PPI (Jahresveränderung)',
            'PIEAMP01EUM661N_yoy': 'EU-PPI (Jahresveränderung)',
            
            # Standardisierte Versionen
            'GDP_standardized': 'BIP USA (standardisiert)',
            'CLVMNACSCAB1GQEU28_standardized': 'BIP Eurozone (standardisiert)',
            'CPIAUCSL_standardized': 'CPI USA (standardisiert)',
            'CP0000EZ19M086NEST_standardized': 'HVPI Eurozone (standardisiert)',
            'PPIACO_standardized': 'PPI USA (standardisiert)',
            'PIEAMP01EUM661N_standardized': 'PPI Eurozone (standardisiert)',
            'FEDFUNDS_standardized': 'Fed Funds Rate (standardisiert)',
            'IR3TIB01EZM156N_standardized': '3M EURIBOR (standardisiert)',
            'UNRATE_standardized': 'Arbeitslosenquote USA (standardisiert)',
            'LRUNTTTTDEQ156S_standardized': 'Arbeitslosenquote DE (standardisiert)',
            'UMCSENT_standardized': 'Verbrauchervertrauen USA (standardisiert)',
            'BSCICP03EZM665S_standardized': 'Verbrauchervertrauen EU (standardisiert)',
            'UNRATE_diff_standardized': 'US-Arbeitslosigkeit Änderung (standardisiert)',
            'LRUNTTTTDEQ156S_diff_standardized': 'DE-Arbeitslosigkeit Änderung (standardisiert)',
            'INTEREST_RATE_SPREAD_standardized': 'Zinsspread (standardisiert)',
            'GDP_growth_standardized': 'US-BIP-Wachstum (standardisiert)',
            'CLVMNACSCAB1GQEU28_growth_standardized': 'EU-BIP-Wachstum (standardisiert)',
            'CPIAUCSL_yoy_standardized': 'US-Inflation YoY (standardisiert)',
            'CP0000EZ19M086NEST_yoy_standardized': 'EU-Inflation YoY (standardisiert)',
            'PPIACO_yoy_standardized': 'US-PPI YoY (standardisiert)',
            'PIEAMP01EUM661N_yoy_standardized': 'EU-PPI YoY (standardisiert)',
        }
        
        # Kategorien für bessere Übersicht
        self.variable_categories = {
            'BIP & Wachstum': [
                'GDP', 'CLVMNACSCAB1GQEU28', 'GDP_growth', 'CLVMNACSCAB1GQEU28_growth'
            ],
            'Inflation': [
                'CPIAUCSL', 'CP0000EZ19M086NEST', 'PPIACO', 'PIEAMP01EUM661N',
                'CPIAUCSL_yoy', 'CP0000EZ19M086NEST_yoy', 'PPIACO_yoy', 'PIEAMP01EUM661N_yoy'
            ],
            'Zinsen': [
                'FEDFUNDS', 'EUR3MTD156N', 'INTEREST_RATE_SPREAD'
            ],
            'Arbeitsmarkt': [
                'UNRATE', 'LRUNTTTTDEQ156S', 'UNRATE_diff', 'LRUNTTTTDEQ156S_diff'
            ],
            'Verbrauchervertrauen': [
                'UMCSENT', 'BSCICP03EZM665S'
            ]
        }
    
    def calculate_comprehensive_statistics(self, event_matrix: pd.DataFrame) -> Dict:
        """
        Berechnet umfassende Statistiken für alle Variablen
        
        Args:
            event_matrix: Ereignismatrix mit allen Variablen
            
        Returns:
            Dictionary mit detaillierten Statistiken
        """
        logger.info("Starte umfassende statistische Analyse")
        
        statistics = {
            'overall_summary': {},
            'category_statistics': {},
            'variable_statistics': {},
            'correlation_analysis': {},
            'time_series_analysis': {}
        }
        
        # 1. Gesamtübersicht
        statistics['overall_summary'] = self._create_overall_summary(event_matrix)
        
        # 2. Statistiken nach Kategorien
        statistics['category_statistics'] = self._create_category_statistics(event_matrix)
        
        # 3. Detaillierte Variablenstatistiken
        statistics['variable_statistics'] = self._create_variable_statistics(event_matrix)
        
        # 4. Korrelationsanalyse
        statistics['correlation_analysis'] = self._create_correlation_analysis(event_matrix)
        
        # 5. Zeitreihenanalyse
        statistics['time_series_analysis'] = self._create_time_series_analysis(event_matrix)
        
        logger.info("Statistische Analyse abgeschlossen")
        return statistics
    
    def _create_overall_summary(self, event_matrix: pd.DataFrame) -> Dict:
        """Erstellt Gesamtübersicht"""
        logger.info("Erstelle Gesamtübersicht")
        
        return {
            'total_events': len(event_matrix),
            'date_range': {
                'start': event_matrix['publication_date'].min().strftime('%Y-%m-%d'),
                'end': event_matrix['publication_date'].max().strftime('%Y-%m-%d'),
                'total_days': (event_matrix['publication_date'].max() - event_matrix['publication_date'].min()).days
            },
            'unique_indicators': event_matrix['indicator'].nunique(),
            'indicator_list': event_matrix['indicator'].unique().tolist(),
            'missing_data_summary': {
                col: event_matrix[col].isna().sum() 
                for col in event_matrix.columns 
                if event_matrix[col].isna().sum() > 0
            }
        }
    
    def _create_category_statistics(self, event_matrix: pd.DataFrame) -> Dict:
        """Erstellt Statistiken nach Kategorien"""
        logger.info("Erstelle Kategoriestatistiken")
        
        category_stats = {}
        
        for category, variables in self.variable_categories.items():
            category_data = {}
            available_vars = [var for var in variables if var in event_matrix.columns]
            
            if available_vars:
                category_data['variables'] = available_vars
                category_data['variable_count'] = len(available_vars)
                
                # Statistiken für verfügbare Variablen
                for var in available_vars:
                    if event_matrix[var].dtype in ['float64', 'int64']:
                        var_stats = self._calculate_basic_statistics(event_matrix[var])
                        category_data[var] = {
                            'german_name': self.variable_names.get(var, var),
                            'statistics': var_stats
                        }
            
            category_stats[category] = category_data
        
        return category_stats
    
    def _create_variable_statistics(self, event_matrix: pd.DataFrame) -> Dict:
        """Erstellt detaillierte Variablenstatistiken"""
        logger.info("Erstelle detaillierte Variablenstatistiken")
        
        var_stats = {}
        
        # Alle numerischen Spalten analysieren
        numeric_cols = event_matrix.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['publication_date'] or col.endswith('_direction'):
                continue
                
            stats = self._calculate_detailed_statistics(event_matrix[col])
            var_stats[col] = {
                'german_name': self.variable_names.get(col, col),
                'data_type': str(event_matrix[col].dtype),
                'statistics': stats,
                'distribution_info': self._analyze_distribution(event_matrix[col])
            }
        
        return var_stats
    
    def _calculate_basic_statistics(self, series: pd.Series) -> Dict:
        """Berechnet Grundstatistiken"""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'Keine gültigen Daten verfügbar'}
        
        return {
            'count': int(len(series_clean)),
            'mean': float(series_clean.mean()),
            'std': float(series_clean.std()),
            'min': float(series_clean.min()),
            'max': float(series_clean.max()),
            'q25': float(series_clean.quantile(0.25)),
            'median': float(series_clean.quantile(0.50)),
            'q75': float(series_clean.quantile(0.75)),
            'missing_values': int(series.isna().sum()),
            'missing_percentage': float(series.isna().sum() / len(series) * 100)
        }
    
    def _calculate_detailed_statistics(self, series: pd.Series) -> Dict:
        """Berechnet detaillierte Statistiken"""
        basic_stats = self._calculate_basic_statistics(series)
        
        if 'error' in basic_stats:
            return basic_stats
        
        series_clean = series.dropna()
        
        # Erweiterte Statistiken
        basic_stats.update({
            'range': float(series_clean.max() - series_clean.min()),
            'iqr': float(series_clean.quantile(0.75) - series_clean.quantile(0.25)),
            'skewness': float(series_clean.skew()),
            'kurtosis': float(series_clean.kurtosis()),
            'variance': float(series_clean.var()),
            'coefficient_of_variation': float(series_clean.std() / abs(series_clean.mean())) if series_clean.mean() != 0 else float('inf'),
            'percentiles': {
                f'p{p}': float(series_clean.quantile(p/100)) 
                for p in [1, 5, 10, 90, 95, 99]
            }
        })
        
        return basic_stats
    
    def _analyze_distribution(self, series: pd.Series) -> Dict:
        """Analysiert die Verteilung der Daten"""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'Keine gültigen Daten verfügbar'}
        
        # Normalverteilungstest (vereinfacht)
        if SCIPY_AVAILABLE:
            try:
                # Shapiro-Wilk Test (nur für kleine Stichproben)
                if len(series_clean) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(series_clean)
                    normality_test = {
                        'test': 'Shapiro-Wilk',
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    }
                else:
                    # Für größere Stichproben: Kolmogorov-Smirnov Test
                    ks_stat, ks_p = stats.kstest(series_clean, 'norm')
                    normality_test = {
                        'test': 'Kolmogorov-Smirnov',
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p),
                        'is_normal': ks_p > 0.05
                    }
            except:
                normality_test = {'error': 'Normalverteilungstest fehlgeschlagen'}
        else:
            normality_test = {'error': 'Scipy nicht verfügbar für Normalverteilungstest'}
        
        return {
            'normality_test': normality_test,
            'outliers': self._detect_outliers(series_clean),
            'zero_values': int((series_clean == 0).sum()),
            'negative_values': int((series_clean < 0).sum()),
            'positive_values': int((series_clean > 0).sum())
        }
    
    def _detect_outliers(self, series: pd.Series) -> Dict:
        """Erkennt Ausreißer mit verschiedenen Methoden"""
        # IQR-Methode
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        iqr_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        
        # Z-Score-Methode (3-Sigma-Regel)
        z_scores = np.abs((series - series.mean()) / series.std())
        zscore_outliers = (z_scores > 3).sum()
        
        return {
            'iqr_method': {
                'count': int(iqr_outliers),
                'percentage': float(iqr_outliers / len(series) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            },
            'zscore_method': {
                'count': int(zscore_outliers),
                'percentage': float(zscore_outliers / len(series) * 100),
                'threshold': 3.0
            }
        }
    
    def _create_correlation_analysis(self, event_matrix: pd.DataFrame) -> Dict:
        """Erstellt Korrelationsanalyse"""
        logger.info("Erstelle Korrelationsanalyse")
        
        # Nur numerische Spalten für Korrelation
        numeric_cols = event_matrix.select_dtypes(include=[np.number]).columns
        indicator_cols = [col for col in numeric_cols if col.endswith('_standardized')]
        
        if len(indicator_cols) < 2:
            return {'error': 'Nicht genügend numerische Variablen für Korrelationsanalyse'}
        
        correlation_matrix = event_matrix[indicator_cols].corr()
        
        # Finde stärkste Korrelationen
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if not np.isnan(corr_value):
                    correlations.append({
                        'variable1': var1,
                        'variable1_german': self.variable_names.get(var1, var1),
                        'variable2': var2,
                        'variable2_german': self.variable_names.get(var2, var2),
                        'correlation': float(corr_value),
                        'abs_correlation': float(abs(corr_value))
                    })
        
        # Sortiere nach absoluter Korrelation
        correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'top_correlations': correlations[:20],  # Top 20 Korrelationen
            'high_correlations': [c for c in correlations if c['abs_correlation'] > 0.7],
            'moderate_correlations': [c for c in correlations if 0.3 < c['abs_correlation'] <= 0.7],
            'low_correlations': [c for c in correlations if c['abs_correlation'] <= 0.3]
        }
    
    def _create_time_series_analysis(self, event_matrix: pd.DataFrame) -> Dict:
        """Erstellt Zeitreihenanalyse"""
        logger.info("Erstelle Zeitreihenanalyse")
        
        # Gruppiere nach Jahren und Monaten
        event_matrix['year'] = event_matrix['publication_date'].dt.year
        event_matrix['month'] = event_matrix['publication_date'].dt.month
        
        # Jährliche Statistiken
        yearly_stats = {}
        for year in sorted(event_matrix['year'].unique()):
            year_data = event_matrix[event_matrix['year'] == year]
            yearly_stats[int(year)] = {
                'event_count': len(year_data),
                'unique_indicators': year_data['indicator'].nunique(),
                'date_range': {
                    'start': year_data['publication_date'].min().strftime('%Y-%m-%d'),
                    'end': year_data['publication_date'].max().strftime('%Y-%m-%d')
                }
            }
        
        # Monatliche Verteilung
        monthly_distribution = event_matrix.groupby('month').size().to_dict()
        monthly_distribution = {int(k): int(v) for k, v in monthly_distribution.items()}
        
        # Indikator-Häufigkeit
        indicator_frequency = event_matrix['indicator'].value_counts().to_dict()
        
        return {
            'yearly_statistics': yearly_stats,
            'monthly_distribution': monthly_distribution,
            'indicator_frequency': indicator_frequency,
            'data_coverage': {
                'total_years': len(yearly_stats),
                'years_with_data': list(yearly_stats.keys()),
                'most_frequent_month': int(max(monthly_distribution, key=monthly_distribution.get)),
                'most_frequent_indicator': max(indicator_frequency, key=indicator_frequency.get)
            }
        }
    
    def create_formatted_report(self, statistics: Dict) -> str:
        """Erstellt einen formatierten Bericht"""
        report = []
        report.append("=" * 100)
        report.append("UMFASSENDE STATISTISCHE ANALYSE ALLER VARIABLEN")
        report.append("=" * 100)
        report.append(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Gesamtübersicht
        overall = statistics['overall_summary']
        report.append("GESAMTÜBERSICHT")
        report.append("-" * 50)
        report.append(f"• Gesamtanzahl Ereignisse: {overall['total_events']:,}")
        report.append(f"• Zeitraum: {overall['date_range']['start']} bis {overall['date_range']['end']}")
        report.append(f"• Anzahl Tage: {overall['date_range']['total_days']:,}")
        report.append(f"• Anzahl Indikatoren: {overall['unique_indicators']}")
        report.append("")
        
        # Kategoriestatistiken
        report.append("STATISTIKEN NACH KATEGORIEN")
        report.append("-" * 50)
        
        for category, data in statistics['category_statistics'].items():
            if 'variables' in data:
                report.append(f"\n📊 {category}")
                report.append(f"   Anzahl Variablen: {data['variable_count']}")
                
                for var in data['variables']:
                    if var in data and 'statistics' in data[var]:
                        stats = data[var]['statistics']
                        if 'error' not in stats:
                            report.append(f"   • {data[var]['german_name']}:")
                            report.append(f"     - Mittelwert: {stats['mean']:.4f}")
                            report.append(f"     - Standardabw.: {stats['std']:.4f}")
                            report.append(f"     - Min/Max: {stats['min']:.4f} / {stats['max']:.4f}")
                            report.append(f"     - Quartile: {stats['q25']:.4f} | {stats['median']:.4f} | {stats['q75']:.4f}")
                            report.append(f"     - Anzahl Werte: {stats['count']:,}")
        
        # Top Korrelationen
        if 'correlation_analysis' in statistics and 'top_correlations' in statistics['correlation_analysis']:
            report.append("\n\nTOP KORRELATIONEN")
            report.append("-" * 50)
            
            for i, corr in enumerate(statistics['correlation_analysis']['top_correlations'][:10], 1):
                report.append(f"{i:2d}. {corr['variable1_german']} ↔ {corr['variable2_german']}: {corr['correlation']:.4f}")
        
        # Zeitreihenanalyse
        if 'time_series_analysis' in statistics:
            ts_data = statistics['time_series_analysis']
            report.append("\n\nZEITREIHENANALYSE")
            report.append("-" * 50)
            report.append(f"• Abgedeckte Jahre: {ts_data['data_coverage']['total_years']}")
            report.append(f"• Häufigster Monat: {ts_data['data_coverage']['most_frequent_month']}")
            report.append(f"• Häufigster Indikator: {ts_data['data_coverage']['most_frequent_indicator']}")
        
        report.append("\n" + "=" * 100)
        report.append("ENDE DES BERICHTS")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def _convert_numpy_types(self, obj):
        """Konvertiert numpy-Typen zu JSON-serialisierbaren Typen"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def save_results(self, statistics: Dict, filename_prefix: str = "variable_statistics"):
        """Speichert die Ergebnisse"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Konvertiere numpy-Typen für JSON-Serialisierung
        json_safe_statistics = self._convert_numpy_types(statistics)
        
        # JSON-Datei
        json_path = f"results/{filename_prefix}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe_statistics, f, indent=4, ensure_ascii=False)
        logger.info(f"Statistiken gespeichert: {json_path}")
        
        # Formatierter Bericht
        report = self.create_formatted_report(statistics)
        report_path = f"results/{filename_prefix}_{timestamp}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Bericht gespeichert: {report_path}")
        
        return json_path, report_path

def main():
    """Hauptfunktion"""
    # Erstelle notwendige Verzeichnisse
    os.makedirs('results', exist_ok=True)
    
    analyzer = VariableStatisticsAnalyzer()
    
    # Lade die neuesten Daten
    try:
        # Finde die neueste Event-Matrix-Datei
        data_files = [f for f in os.listdir('data') if f.startswith('event_matrix_') and f.endswith('.csv')]
        if not data_files:
            logger.error("Keine Event-Matrix-Dateien gefunden. Führen Sie zuerst main.py aus.")
            return
        
        latest_file = sorted(data_files)[-1]
        logger.info(f"Lade Daten aus: {latest_file}")
        
        event_matrix = analyzer.input_module.load_data(latest_file)
        
        if event_matrix.empty:
            logger.error("Event-Matrix ist leer")
            return
        
        # Führe statistische Analyse durch
        statistics = analyzer.calculate_comprehensive_statistics(event_matrix)
        
        # Speichere Ergebnisse
        json_path, report_path = analyzer.save_results(statistics)
        
        # Zeige Zusammenfassung
        print("\n" + "=" * 80)
        print("STATISTISCHE ANALYSE ABGESCHLOSSEN")
        print("=" * 80)
        print(f"JSON-Datei: {json_path}")
        print(f"Bericht: {report_path}")
        print(f"Analysierte Ereignisse: {statistics['overall_summary']['total_events']:,}")
        print(f"Analysierte Variablen: {len(statistics['variable_statistics'])}")
        print("=" * 80)
        
        # Zeige kurze Zusammenfassung der wichtigsten Statistiken
        print("\nWICHTIGSTE ERKENNTNISSE:")
        print("-" * 40)
        
        # Top 5 Korrelationen
        if 'top_correlations' in statistics['correlation_analysis']:
            print("\nTop 5 Korrelationen:")
            for i, corr in enumerate(statistics['correlation_analysis']['top_correlations'][:5], 1):
                print(f"{i}. {corr['variable1_german']} ↔ {corr['variable2_german']}: {corr['correlation']:.4f}")
        
    except Exception as e:
        logger.error(f"Fehler bei der Analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()