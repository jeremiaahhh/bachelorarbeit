"""
Input-Modul: Erhebung, Bereinigung und Harmonisierung heterogener Datenströme
Implementiert die automatisierte Erfassung der Rohdaten sowie deren methodisch konsistente Aufbereitung.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import logging

from config import Config

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Klasse zur Datenerhebung aus verschiedenen Quellen"""
    
    def __init__(self):
        """Initialisiert die Datenquellen"""
        self.fred = Fred(api_key=Config.FRED_API_KEY)
        self.forex_symbol = "EURUSD=X"
        
    def fetch_macroeconomic_data(self, start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """
        Lädt makroökonomische Daten aus FRED
        
        Args:
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD'
            
        Returns:
            Dictionary mit Indikator-Serien
        """
        logger.info(f"Lade makroökonomische Daten von {start_date} bis {end_date}")
        
        macro_data = {}
        
        for indicator, description in Config.FRED_INDICATORS.items():
            try:
                logger.info(f"Lade {indicator}: {description}")
                data = self.fred.get_series(indicator, start_date, end_date)
                
                # Validiere die geladenen Daten
                if data is not None and len(data) > 0:
                    # Entferne ungültige Zeitstempel (weniger restriktiv)
                    valid_data = data[data.index >= pd.Timestamp('1990-01-01')]
                    if len(valid_data) > 0:
                        # Normalisiere Zeitstempel (entferne timezone info)
                        valid_data.index = valid_data.index.tz_localize(None)
                        macro_data[indicator] = valid_data
                        logger.info(f"Erfolgreich geladen: {len(valid_data)} gültige Datenpunkte für {indicator}")
                    else:
                        logger.warning(f"Keine gültigen Daten für {indicator} im Zeitraum")
                else:
                    logger.warning(f"Keine Daten für {indicator} gefunden")
                    
            except Exception as e:
                logger.warning(f"Fehler beim Laden von {indicator}: {e}")
                continue
        
        # Berechne zusätzliche abgeleitete Indikatoren
        macro_data = self._calculate_derived_indicators(macro_data)
                
        return macro_data
    
    def _calculate_derived_indicators(self, macro_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Berechnet abgeleitete Indikatoren wie Differenzen
        
        Args:
            macro_data: Dictionary mit Rohdaten
            
        Returns:
            Dictionary mit erweiterten Daten
        """
        logger.info("Berechne abgeleitete Indikatoren")
        
        # Arbeitslosenquote-Differenzen (Monatsdifferenzen)
        if 'UNRATE' in macro_data and len(macro_data['UNRATE']) > 1:
            macro_data['UNRATE_diff'] = macro_data['UNRATE'].diff()
            logger.info("UNRATE-Differenz berechnet")
        
        if 'LRUNTTTTDEQ156S' in macro_data and len(macro_data['LRUNTTTTDEQ156S']) > 1:
            macro_data['LRUNTTTTDEQ156S_diff'] = macro_data['LRUNTTTTDEQ156S'].diff()
            logger.info("LRUNTTTTDEQ156S-Differenz berechnet")
        
        # Zinsspread zwischen USA und Eurozone (nur Kurzfristzinsen)
        try:
            if 'FEDFUNDS' in macro_data and 'IR3TIB01EZM156N' in macro_data:
                macro_data['INTEREST_RATE_SPREAD'] = macro_data['FEDFUNDS'] - macro_data['IR3TIB01EZM156N']
                logger.info("Zinsspread (Kurzfristzinsen) USA-Eurozone berechnet")
            else:
                logger.warning("Zinsspread konnte nicht berechnet werden: fehlende Kurzfristzinsreihen (FEDFUNDS oder IR3TIB01EZM156N)")
        except Exception as e:
            logger.warning(f"Fehler bei Zinsspread-Berechnung: {e}")
        
        # BIP-Wachstumsraten (jährliche Veränderung)
        if 'GDP' in macro_data and len(macro_data['GDP']) > 4:
            macro_data['GDP_growth'] = macro_data['GDP'].pct_change(periods=4) * 100
            logger.info("US-BIP-Wachstumsrate berechnet")
        
        if 'CLVMNACSCAB1GQEU28' in macro_data and len(macro_data['CLVMNACSCAB1GQEU28']) > 4:
            macro_data['CLVMNACSCAB1GQEU28_growth'] = macro_data['CLVMNACSCAB1GQEU28'].pct_change(periods=4) * 100
            logger.info("Eurozone-BIP-Wachstumsrate berechnet")
        
        # Inflation-Differenzen (Jahresveränderung)
        if 'CPIAUCSL' in macro_data and len(macro_data['CPIAUCSL']) > 12:
            macro_data['CPIAUCSL_yoy'] = macro_data['CPIAUCSL'].pct_change(periods=12) * 100
            logger.info("US-CPI Jahresveränderung berechnet")
        
        if 'CP0000EZ19M086NEST' in macro_data and len(macro_data['CP0000EZ19M086NEST']) > 12:
            macro_data['CP0000EZ19M086NEST_yoy'] = macro_data['CP0000EZ19M086NEST'].pct_change(periods=12) * 100
            logger.info("Eurozone-HVPI Jahresveränderung berechnet")
        
        # PPI-Differenzen (Jahresveränderung)
        if 'PPIACO' in macro_data and len(macro_data['PPIACO']) > 12:
            macro_data['PPIACO_yoy'] = macro_data['PPIACO'].pct_change(periods=12) * 100
            logger.info("US-PPI Jahresveränderung berechnet")
        
        if 'PIEAMP01EUM661N' in macro_data and len(macro_data['PIEAMP01EUM661N']) > 12:
            macro_data['PIEAMP01EUM661N_yoy'] = macro_data['PIEAMP01EUM661N'].pct_change(periods=12) * 100
            logger.info("Eurozone-PPI Jahresveränderung berechnet")
        
        return macro_data
    
    def fetch_forex_data(self, start_date: str, end_date: str) -> pd.Series:
        """
        Lädt EUR/USD Wechselkursdaten
        
        Args:
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD'
            
        Returns:
            Series mit täglichen Schlusskursen
        """
        logger.info(f"Lade EUR/USD Daten von {start_date} bis {end_date}")
        
        try:
            ticker = yf.Ticker(self.forex_symbol)
            data = ticker.history(start=start_date, end=end_date)
            forex_series = data['Close']
            
            # Konvertiere zu timezone-naive Zeitstempel
            forex_series.index = forex_series.index.tz_localize(None)
            
            logger.info(f"Erfolgreich geladen: {len(forex_series)} Handelstage")
            return forex_series
        except Exception as e:
            logger.error(f"Fehler beim Laden der Forex-Daten: {e}")
            raise

class DataCleaner:
    """Klasse zur Datenbereinigung und -aufbereitung"""
    
    @staticmethod
    def clean_macroeconomic_data(macro_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Bereinigt makroökonomische Daten
        
        Args:
            macro_data: Dictionary mit Rohdaten
            
        Returns:
            Dictionary mit bereinigten Daten
        """
        logger.info("Starte Datenbereinigung (AS-OF/LOCF für Makrodaten)")
        
        cleaned_data = {}
        
        for indicator, series in macro_data.items():
            if series.empty:
                logger.warning(f"Leere Serie für {indicator}")
                continue
                
            # Entferne Duplikate
            series = series[~series.index.duplicated(keep='first')]
            
            # Identifiziere und behandle Ausreißer (3-Sigma-Regel)
            mean_val = series.mean()
            std_val = series.std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            
            outliers = (series < lower_bound) | (series > upper_bound)
            if outliers.sum() > 0:
                logger.info(f"{outliers.sum()} Ausreißer in {indicator} identifiziert")
                # Ersetze Ausreißer durch NaN für spätere Interpolation
                series[outliers] = np.nan
            
            # AS-OF/LOCF: Keine lineare oder Spline-Interpolation.
            # Trage ausschließlich den zuletzt bekannten Veröffentlichungswert fort.
            # Führende NaNs bleiben erhalten (vor der ersten Veröffentlichung keine Information).
            series = series.ffill()
            
            cleaned_data[indicator] = series
            
        logger.info("Datenbereinigung abgeschlossen")
        return cleaned_data
    
    @staticmethod
    def clean_forex_data(forex_series: pd.Series) -> pd.Series:
        """
        Bereinigt Forex-Daten
        
        Args:
            forex_series: Rohdaten der Wechselkurse
            
        Returns:
            Bereinigte Wechselkursdaten
        """
        logger.info("Bereinige Forex-Daten")
        
        # Entferne Duplikate
        forex_series = forex_series[~forex_series.index.duplicated(keep='first')]
        
        # Entferne negative oder null Werte
        forex_series = forex_series[forex_series > 0]
        
        # Forex ist tagesgenau; fehlende Handelstage (Wochenenden/Feiertage) werden NICHT
        # auf Makrovariablen interpoliert. Für Preiszeitreihen ist eine lineare Interpolation
        # über kurze Lücken üblich; die As-of/LOCF-Restriktion gilt für Prädiktoren,
        # nicht für die Zielpreisreihe. Daher belassen wir dies unverändert.
        
        logger.info(f"Forex-Daten bereinigt: {len(forex_series)} Datenpunkte")
        return forex_series

class DataHarmonizer:
    """Klasse zur Harmonisierung heterogener Datenfrequenzen"""
    
    def __init__(self):
        """Initialisiert den Harmonizer"""
        self.forecast_horizons = Config.FORECAST_HORIZONS
        
    def create_event_matrix(self, macro_data: Dict[str, pd.Series], 
                           forex_data: pd.Series) -> pd.DataFrame:
        """
        Erstellt eine harmonisierte Ereignismatrix
        
        Args:
            macro_data: Bereinigte makroökonomische Daten
            forex_data: Bereinigte Forex-Daten
            
        Returns:
            DataFrame mit harmonisierten Ereignissen
        """
        logger.info("Erstelle harmonisierte Ereignismatrix")
        
        events = []
        total_attempts = 0
        successful_events = 0
        
        for indicator, series in macro_data.items():
            logger.info(f"Verarbeite {indicator}: {len(series)} Datenpunkte")
            for date, value in series.items():
                total_attempts += 1
                # Berechne Kursentwicklung für alle Prognosehorizonte
                event_data = self._calculate_forecast_returns(
                    date, value, indicator, forex_data
                )
                if event_data:
                    events.append(event_data)
                    successful_events += 1
        
        logger.info(f"Ereignisverarbeitung abgeschlossen: {successful_events}/{total_attempts} erfolgreich")
        
        if not events:
            logger.warning("Keine Ereignisse gefunden")
            return pd.DataFrame()
        
        event_matrix = pd.DataFrame(events)
        event_matrix = event_matrix.sort_values('publication_date')
        
        logger.info(f"Ereignismatrix erstellt: {len(event_matrix)} Ereignisse")
        return event_matrix
    
    def _calculate_forecast_returns(self, publication_date: datetime, 
                                  indicator_value: float, indicator_name: str,
                                  forex_data: pd.Series) -> Optional[Dict]:
        """
        Berechnet die Kursentwicklung für verschiedene Prognosehorizonte
        
        Args:
            publication_date: Veröffentlichungsdatum
            indicator_value: Veröffentlichter Wert
            indicator_name: Name des Indikators
            forex_data: Wechselkursdaten
            
        Returns:
            Dictionary mit Ereignisdaten oder None
        """
        try:
            # Validiere das Veröffentlichungsdatum (weniger restriktiv)
            if pd.isna(publication_date) or publication_date < pd.Timestamp('1990-01-01'):
                logger.warning(f"Ungültiges Veröffentlichungsdatum für {indicator_name}: {publication_date}")
                return None
            
            # Finde den Kurs vor der Veröffentlichung (letzter Handelskurs ≤ t)
            pre_publication_date = publication_date - timedelta(days=1)
            while pre_publication_date not in forex_data.index:
                pre_publication_date -= timedelta(days=1)
                # Verhindere Endlosschleife
                if pre_publication_date < forex_data.index.min():
                    logger.warning(f"Kein gültiger Vorkurs gefunden für {indicator_name} am {publication_date}")
                    return None
            
            pre_rate = forex_data[pre_publication_date]
            
            event_data = {
                'publication_date': publication_date,
                'indicator': indicator_name,
                'indicator_value': indicator_value,
                'pre_rate': pre_rate
            }
            
            # Berechne Kursentwicklung für jeden Horizont
            for horizon_name, horizon_days in self.forecast_horizons.items():
                post_date = publication_date + timedelta(days=horizon_days)
                
                # Validiere das Post-Datum (weniger restriktiv)
                if pd.isna(post_date) or post_date < pd.Timestamp('1990-01-01'):
                    logger.warning(f"Ungültiges Post-Datum für {indicator_name} {horizon_name}: {post_date}")
                    continue
                
                # Finde den nächsten verfügbaren Kurs (≥ t+h)
                attempts = 0
                while post_date not in forex_data.index:
                    post_date += timedelta(days=1)
                    attempts += 1
                    # Verhindere Endlosschleife
                    if post_date > forex_data.index.max() or attempts > 30:
                        logger.warning(f"Kein gültiger Nachkurs gefunden für {indicator_name} {horizon_name} nach {attempts} Versuchen")
                        break
                
                # Prüfe ob gültiger Kurs gefunden wurde
                if post_date in forex_data.index:
                    post_rate = forex_data[post_date]
                    
                    # Validiere die Kurswerte
                    if pd.isna(pre_rate) or pd.isna(post_rate) or pre_rate <= 0 or post_rate <= 0:
                        logger.warning(f"Ungültige Kurswerte für {indicator_name} {horizon_name}: pre={pre_rate}, post={post_rate}")
                        continue
                    
                    # Berechne relative Kursänderung (entsprechend der Methodik)
                    rate_change = ((post_rate - pre_rate) / pre_rate) * 100
                    
                    # Klassifiziere die Bewegung
                    if rate_change >= Config.MOVEMENT_THRESHOLD:
                        direction = 1  # Long
                    elif rate_change <= -Config.MOVEMENT_THRESHOLD:
                        direction = 0  # Short
                    else:
                        direction = np.nan  # Neutral
                    
                    event_data[f'{horizon_name}_post_rate'] = post_rate
                    event_data[f'{horizon_name}_rate_change'] = rate_change
                    event_data[f'{horizon_name}_direction'] = direction
            
            return event_data
            
        except Exception as e:
            logger.warning(f"Fehler bei der Berechnung für {indicator_name} am {publication_date}: {e}")
            return None
    
    def standardize_features(self, event_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Standardisiert die Inputvariablen mittels z-Transformation
        
        Args:
            event_matrix: Ereignismatrix
            
        Returns:
            DataFrame mit standardisierten Features
        """
        logger.info("Standardisiere Features")
        
        # Identifiziere numerische Spalten für Standardisierung
        numeric_columns = event_matrix.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns 
                         if col not in ['pre_rate'] + 
                         [f'{h}_post_rate' for h in self.forecast_horizons.keys()] +
                         [f'{h}_rate_change' for h in self.forecast_horizons.keys()] +
                         [f'{h}_direction' for h in self.forecast_horizons.keys()]]
        
        # Z-Transformation
        for col in feature_columns:
            if event_matrix[col].notna().sum() > 0:
                mean_val = event_matrix[col].mean()
                std_val = event_matrix[col].std()
                if std_val > 0:
                    event_matrix[f'{col}_standardized'] = (event_matrix[col] - mean_val) / std_val
                else:
                    event_matrix[f'{col}_standardized'] = 0
        
        logger.info(f"Features standardisiert: {len(feature_columns)} Spalten")
        return event_matrix

class InputModule:
    """Hauptklasse des Input-Moduls"""
    
    def __init__(self):
        """Initialisiert das Input-Modul"""
        self.collector = DataCollector()
        self.cleaner = DataCleaner()
        self.harmonizer = DataHarmonizer()
        
    def process_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Vollständige Datenverarbeitung
        
        Args:
            start_date: Startdatum
            end_date: Enddatum
            
        Returns:
            Harmonisierte und aufbereitete Ereignismatrix
        """
        logger.info("Starte vollständige Datenverarbeitung")
        
        # 1. Datenerhebung
        macro_data = self.collector.fetch_macroeconomic_data(start_date, end_date)
        forex_data = self.collector.fetch_forex_data(start_date, end_date)
        
        # 2. Datenbereinigung
        cleaned_macro = self.cleaner.clean_macroeconomic_data(macro_data)
        cleaned_forex = self.cleaner.clean_forex_data(forex_data)
        
        # 3. Harmonisierung
        event_matrix = self.harmonizer.create_event_matrix(cleaned_macro, cleaned_forex)
        
        # 4. Standardisierung
        if not event_matrix.empty:
            event_matrix = self.harmonizer.standardize_features(event_matrix)
        
        logger.info("Datenverarbeitung abgeschlossen")
        return event_matrix
    
    def save_data(self, event_matrix: pd.DataFrame, filename: str):
        """
        Speichert die verarbeiteten Daten
        
        Args:
            event_matrix: Ereignismatrix
            filename: Dateiname
        """
        filepath = f"{Config.DATA_DIR}/{filename}"
        event_matrix.to_csv(filepath, index=False)
        logger.info(f"Daten gespeichert: {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Lädt gespeicherte Daten
        
        Args:
            filename: Dateiname
            
        Returns:
            Geladene Ereignismatrix
        """
        filepath = f"{Config.DATA_DIR}/{filename}"
        event_matrix = pd.read_csv(filepath)
        event_matrix['publication_date'] = pd.to_datetime(event_matrix['publication_date'])
        logger.info(f"Daten geladen: {filepath}")
        return event_matrix 