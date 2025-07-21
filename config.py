"""
Konfigurationsdatei für das Devisenhandel-Vorhersagesystem
"""
import os
from datetime import datetime
from dotenv import load_dotenv

# Lade Umgebungsvariablen (mit Fehlerbehandlung)
try:
    load_dotenv()
except Exception as e:
    print(f"Warnung: .env-Datei konnte nicht geladen werden: {e}")

class Config:
    """Zentrale Konfigurationsklasse für das System"""
    
    # API-Schlüssel
    FRED_API_KEY = os.getenv('FRED_API_KEY', '')
    
    # Zeiträume
    TRAINING_START = '2003-01-01'
    TRAINING_END = '2019-12-31'
    TEST_START = '2020-01-01'
    TEST_END = '2025-12-31'
    
    # Prognosehorizonte (Handelstage)
    FORECAST_HORIZONS = {
        'daily': 1,
        'weekly': 5,
        'monthly': 20,
        'yearly': 250
    }
    
    # Schwellenwerte für Klassifikation
    CLASSIFICATION_THRESHOLD = 0.6378
    MOVEMENT_THRESHOLD = 0  # 0.25% für signifikante Bewegungen
    
    # Intensitätsklassen
    INTENSITY_CLASSES = {
        'weak': (0.50, 0.60),
        'moderate': (0.60, 0.75),
        'strong': (0.75, 1.00)
    }
    
    # FRED-Indikatoren
    FRED_INDICATORS = {
        # Bruttoinlandsprodukt (BIP)
        'GDP': 'Bruttoinlandsprodukt (BIP) USA, vierteljährlich',
        'CLVMNACSCAB1GQEU28': 'Bruttoinlandsprodukt (BIP) Eurozone, vierteljährlich',
        
        # Verbraucherpreisindex (Inflation)
        'CPIAUCSL': 'Verbraucherpreisindex (CPI) USA, monatlich',
        'CP0000EZ19M086NEST': 'Verbraucherpreisindex (HVPI) Eurozone, monatlich',
        
        # Produzentenpreisindex (PPI)
        'PPIACO': 'Produzentenpreisindex (PPI) USA',
        'PIEAMP01EUM661N': 'Produzentenpreisindex (PPI) Eurozone',
        
        # Zinsen (Kurzfrist)
        'FEDFUNDS': 'Effektiver Fed Funds Rate (USA, monatlich)',
        'IR3TIB01EZM156N': '3-Monats EURIBOR (Euro Area, monatlich)',
        
        # Arbeitsmarkt
        'UNRATE': 'Arbeitslosenquote USA, monatlich',
        'LRUNTTTTDEQ156S': 'Arbeitslosenquote Deutschland, monatlich',
        
        # Verbrauchervertrauen
        'UMCSENT': 'Verbrauchervertrauen USA (Michigan Consumer Sentiment Index)',
        'BSCICP03EZM665S': 'Verbrauchervertrauen Eurozone (Consumer Confidence Indicator)'
    }
    
    # Datenpfade
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    PLOTS_DIR = 'plots'
    
    # Modellparameter
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Visualisierung
    FIGURE_SIZE = (12, 8)
    DPI = 300
    
    # Umfassende Variablen-Dokumentation
    VARIABLE_SUMMARY = {
        # ===== INPUT VARIABLEN =====
        'input_variables': {
            'publication_date': {
                'description': 'Veröffentlichungsdatum der makroökonomischen Indikatoren',
                'data_type': 'datetime',
                'source': 'FRED API',
                'expected_range': '2003-01-01 bis 2025-12-31',
                'missing_values': 'Keine erwartet',
                'notes': 'Hauptzeitstempel für alle Ereignisse'
            },
            'indicator': {
                'description': 'Name des makroökonomischen Indikators',
                'data_type': 'categorical',
                'source': 'FRED API',
                'categories': list(FRED_INDICATORS.keys()),
                'missing_values': 'Keine erwartet',
                'notes': 'Kategorische Variable für Indikatoridentifikation'
            },
            'indicator_value': {
                'description': 'Rohwert des makroökonomischen Indikators',
                'data_type': 'float',
                'source': 'FRED API',
                'expected_range': 'Variiert je nach Indikator',
                'missing_values': 'Mögliche NaN-Werte bei fehlenden Daten',
                'notes': 'Ursprünglicher Wert vor Standardisierung'
            },
            'indicator_value_standardized': {
                'description': 'Z-standardisierter Wert des Indikators',
                'data_type': 'float',
                'source': 'Berechnet aus indicator_value',
                'expected_range': 'Typisch -3 bis +3 (99.7% der Werte)',
                'missing_values': 'NaN bei fehlenden Rohdaten',
                'notes': 'Standardisierte Version für Modelltraining'
            }
        },
        
        # ===== ABGELEITETE INDIKATOREN =====
        'derived_indicators': {
            'UNRATE_diff': {
                'description': 'Monatsdifferenz der US-Arbeitslosenquote',
                'data_type': 'float',
                'source': 'Berechnet aus UNRATE',
                'expected_range': 'Typisch -1.0 bis +1.0 Prozentpunkte',
                'missing_values': 'NaN im ersten Monat',
                'notes': 'Zeigt Änderungsrichtung der Arbeitslosigkeit'
            },
            'LRUNTTTTDEQ156S_diff': {
                'description': 'Monatsdifferenz der deutschen Arbeitslosenquote',
                'data_type': 'float',
                'source': 'Berechnet aus LRUNTTTTDEQ156S',
                'expected_range': 'Typisch -1.0 bis +1.0 Prozentpunkte',
                'missing_values': 'NaN im ersten Monat',
                'notes': 'Zeigt Änderungsrichtung der deutschen Arbeitslosigkeit'
            },
            'INTEREST_RATE_SPREAD': {
                'description': 'Zinsspread zwischen USA und Eurozone (Kurzfristzinsen)',
                'data_type': 'float',
                'source': 'Berechnet aus FEDFUNDS - IR3TIB01EZM156N',
                'expected_range': 'Historisch -5.0 bis +5.0 Prozentpunkte',
                'missing_values': 'NaN wenn einer der Zinssätze fehlt',
                'notes': 'Wichtiger Indikator für Währungsstärke'
            }
        },
        
        # ===== FEATURE-VARIABLEN =====
        'feature_variables': {
            'year': {
                'description': 'Jahr der Veröffentlichung',
                'data_type': 'integer',
                'source': 'Extrahierte aus publication_date',
                'expected_range': '2003 bis 2025',
                'missing_values': 'Keine erwartet',
                'notes': 'Zeitbasierte Feature für Saisonalität'
            },
            'month': {
                'description': 'Monat der Veröffentlichung',
                'data_type': 'integer',
                'source': 'Extrahierte aus publication_date',
                'expected_range': '1 bis 12',
                'missing_values': 'Keine erwartet',
                'notes': 'Zeitbasierte Feature für Saisonalität'
            },
            'quarter': {
                'description': 'Quartal der Veröffentlichung',
                'data_type': 'integer',
                'source': 'Extrahierte aus publication_date',
                'expected_range': '1 bis 4',
                'missing_values': 'Keine erwartet',
                'notes': 'Zeitbasierte Feature für Quartalseffekte'
            },
            'day_of_week': {
                'description': 'Wochentag der Veröffentlichung (0=Montag, 6=Sonntag)',
                'data_type': 'integer',
                'source': 'Extrahierte aus publication_date',
                'expected_range': '0 bis 6',
                'missing_values': 'Keine erwartet',
                'notes': 'Zeitbasierte Feature für Wocheneffekte'
            }
        },
        
        # ===== LAG-FEATURES =====
        'lag_features': {
            'indicator_lag1': {
                'description': 'Standardisierter Indikatorwert mit Lag 1',
                'data_type': 'float',
                'source': 'Berechnet aus indicator_value_standardized',
                'expected_range': 'Typisch -3 bis +3',
                'missing_values': 'NaN im ersten Datenpunkt',
                'notes': 'Vorheriger Wert für Trenderkennung'
            },
            'indicator_lag2': {
                'description': 'Standardisierter Indikatorwert mit Lag 2',
                'data_type': 'float',
                'source': 'Berechnet aus indicator_value_standardized',
                'expected_range': 'Typisch -3 bis +3',
                'missing_values': 'NaN in den ersten beiden Datenpunkten',
                'notes': 'Vor-vorheriger Wert für Trenderkennung'
            }
        },
        
        # ===== ROLLING-STATISTIKEN =====
        'rolling_features': {
            'indicator_rolling_mean_5': {
                'description': 'Rolling-Mittelwert über 5 Perioden',
                'data_type': 'float',
                'source': 'Berechnet aus indicator_value_standardized',
                'expected_range': 'Typisch -3 bis +3',
                'missing_values': 'NaN in den ersten 4 Perioden',
                'notes': 'Gleitender Durchschnitt für Trendglättung'
            },
            'indicator_rolling_std_5': {
                'description': 'Rolling-Standardabweichung über 5 Perioden',
                'data_type': 'float',
                'source': 'Berechnet aus indicator_value_standardized',
                'expected_range': 'Typisch 0 bis 2',
                'missing_values': 'NaN in den ersten 4 Perioden',
                'notes': 'Volatilitätsmaß über Zeitfenster'
            }
        },
        
        # ===== DUMMY-VARIABLEN =====
        'dummy_variables': {
            'indicator_dummies': {
                'description': 'One-Hot-Encoded Dummy-Variablen für jeden Indikator',
                'data_type': 'binary (0/1)',
                'source': 'Berechnet aus indicator',
                'expected_range': '0 oder 1',
                'missing_values': 'Keine erwartet',
                'notes': 'Kategorische Kodierung für Modelltraining'
            }
        },
        
        # ===== ZIELVARIABLEN =====
        'target_variables': {
            'daily_direction': {
                'description': 'Richtung der EUR/USD Kursbewegung (1 Tag)',
                'data_type': 'binary (0/1)',
                'source': 'Berechnet aus EUR/USD Kursdaten',
                'categories': {0: 'Short (Fallen)', 1: 'Long (Steigen)'},
                'threshold': f'{MOVEMENT_THRESHOLD}%',
                'missing_values': 'NaN bei fehlenden Kursdaten',
                'notes': 'Hauptzielvariable für 1-Tages-Prognose'
            },
            'weekly_direction': {
                'description': 'Richtung der EUR/USD Kursbewegung (5 Tage)',
                'data_type': 'binary (0/1)',
                'source': 'Berechnet aus EUR/USD Kursdaten',
                'categories': {0: 'Short (Fallen)', 1: 'Long (Steigen)'},
                'threshold': f'{MOVEMENT_THRESHOLD}%',
                'missing_values': 'NaN bei fehlenden Kursdaten',
                'notes': 'Zielvariable für 1-Wochen-Prognose'
            },
            'monthly_direction': {
                'description': 'Richtung der EUR/USD Kursbewegung (20 Tage)',
                'data_type': 'binary (0/1)',
                'source': 'Berechnet aus EUR/USD Kursdaten',
                'categories': {0: 'Short (Fallen)', 1: 'Long (Steigen)'},
                'threshold': f'{MOVEMENT_THRESHOLD}%',
                'missing_values': 'NaN bei fehlenden Kursdaten',
                'notes': 'Zielvariable für 1-Monats-Prognose'
            },
            'yearly_direction': {
                'description': 'Richtung der EUR/USD Kursbewegung (250 Tage)',
                'data_type': 'binary (0/1)',
                'source': 'Berechnet aus EUR/USD Kursdaten',
                'categories': {0: 'Short (Fallen)', 1: 'Long (Steigen)'},
                'threshold': f'{MOVEMENT_THRESHOLD}%',
                'missing_values': 'NaN bei fehlenden Kursdaten',
                'notes': 'Zielvariable für 1-Jahres-Prognose'
            }
        },
        
        # ===== KURSÄNDERUNGSVARIABLEN =====
        'rate_change_variables': {
            'daily_rate_change': {
                'description': 'Prozentuale Kursänderung EUR/USD (1 Tag)',
                'data_type': 'float',
                'source': 'Berechnet aus EUR/USD Kursdaten',
                'expected_range': 'Typisch -5% bis +5%',
                'missing_values': 'NaN bei fehlenden Kursdaten',
                'notes': 'Kontinuierliche Zielvariable'
            },
            'weekly_rate_change': {
                'description': 'Prozentuale Kursänderung EUR/USD (5 Tage)',
                'data_type': 'float',
                'source': 'Berechnet aus EUR/USD Kursdaten',
                'expected_range': 'Typisch -10% bis +10%',
                'missing_values': 'NaN bei fehlenden Kursdaten',
                'notes': 'Kontinuierliche Zielvariable'
            },
            'monthly_rate_change': {
                'description': 'Prozentuale Kursänderung EUR/USD (20 Tage)',
                'data_type': 'float',
                'source': 'Berechnet aus EUR/USD Kursdaten',
                'expected_range': 'Typisch -20% bis +20%',
                'missing_values': 'NaN bei fehlenden Kursdaten',
                'notes': 'Kontinuierliche Zielvariable'
            },
            'yearly_rate_change': {
                'description': 'Prozentuale Kursänderung EUR/USD (250 Tage)',
                'data_type': 'float',
                'source': 'Berechnet aus EUR/USD Kursdaten',
                'expected_range': 'Typisch -50% bis +50%',
                'missing_values': 'NaN bei fehlenden Kursdaten',
                'notes': 'Kontinuierliche Zielvariable'
            }
        },
        
        # ===== VORHERSAGEVARIABLEN =====
        'prediction_variables': {
            'daily_prediction': {
                'description': 'Vorhersage der Kursrichtung (1 Tag)',
                'data_type': 'binary (0/1)',
                'source': 'Logistische Regression',
                'categories': {0: 'Short', 1: 'Long'},
                'threshold': CLASSIFICATION_THRESHOLD,
                'missing_values': 'NaN bei fehlenden Features',
                'notes': 'Modellvorhersage basierend auf Wahrscheinlichkeit'
            },
            'daily_probability': {
                'description': 'Wahrscheinlichkeit für Long-Position (1 Tag)',
                'data_type': 'float',
                'source': 'Logistische Regression',
                'expected_range': '0.0 bis 1.0',
                'missing_values': 'NaN bei fehlenden Features',
                'notes': 'Rohwahrscheinlichkeit aus Modell'
            },
            'daily_intensity': {
                'description': 'Intensitätsklasse der Vorhersage (1 Tag)',
                'data_type': 'categorical',
                'source': 'Berechnet aus daily_probability',
                'categories': list(INTENSITY_CLASSES.keys()),
                'missing_values': 'NaN bei fehlenden Wahrscheinlichkeiten',
                'notes': 'Klassifikation der Vorhersagestärke'
            }
        },
        
        # ===== EVALUATIONSVARIABLEN =====
        'evaluation_variables': {
            'daily_correct': {
                'description': 'Korrektheit der Vorhersage (1 Tag)',
                'data_type': 'boolean',
                'source': 'Vergleich daily_prediction mit daily_direction',
                'categories': {True: 'Korrekt', False: 'Falsch'},
                'missing_values': 'NaN bei fehlenden tatsächlichen Werten',
                'notes': 'Performance-Metrik für Modellbewertung'
            },
            'accuracy': {
                'description': 'Gesamtgenauigkeit der Vorhersagen',
                'data_type': 'float',
                'source': 'Berechnet aus korrekten Vorhersagen',
                'expected_range': '0.0 bis 1.0',
                'missing_values': 'Keine erwartet',
                'notes': 'Anteil korrekter Vorhersagen'
            },
            'precision': {
                'description': 'Präzision der Long-Vorhersagen',
                'data_type': 'float',
                'source': 'Berechnet aus True Positives / (True Positives + False Positives)',
                'expected_range': '0.0 bis 1.0',
                'missing_values': 'Keine erwartet',
                'notes': 'Anteil korrekter Long-Vorhersagen'
            },
            'recall': {
                'description': 'Recall der Long-Vorhersagen',
                'data_type': 'float',
                'source': 'Berechnet aus True Positives / (True Positives + False Negatives)',
                'expected_range': '0.0 bis 1.0',
                'missing_values': 'Keine erwartet',
                'notes': 'Anteil erfasster Long-Ereignisse'
            },
            'f1_score': {
                'description': 'F1-Score (Harmonisches Mittel aus Precision und Recall)',
                'data_type': 'float',
                'source': 'Berechnet aus 2 * (precision * recall) / (precision + recall)',
                'expected_range': '0.0 bis 1.0',
                'missing_values': 'Keine erwartet',
                'notes': 'Balanced Performance-Metrik'
            },
            'roc_auc': {
                'description': 'ROC-AUC Score',
                'data_type': 'float',
                'source': 'Berechnet aus ROC-Kurve',
                'expected_range': '0.0 bis 1.0 (0.5 = Zufall)',
                'missing_values': 'Keine erwartet',
                'notes': 'Area Under ROC Curve'
            }
        }
    }
    
    @classmethod
    def create_directories(cls):
        """Erstellt notwendige Verzeichnisse"""
        directories = [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.PLOTS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_variable_summary(cls) -> dict:
        """
        Gibt eine Zusammenfassung aller verwendeten Variablen zurück
        
        Returns:
            Dictionary mit Variablenübersicht
        """
        return cls.VARIABLE_SUMMARY
    
    @classmethod
    def print_variable_summary(cls):
        """
        Druckt eine formatierte Übersicht aller Variablen
        """
        print("=" * 80)
        print("VARIABLEN-ÜBERSICHT DES DEVISENHANDEL-VORHERSAGESYSTEMS")
        print("=" * 80)
        
        for category, variables in cls.VARIABLE_SUMMARY.items():
            print(f"\n📊 {category.upper().replace('_', ' ')}")
            print("-" * 60)
            
            for var_name, var_info in variables.items():
                print(f"\n🔹 {var_name}")
                print(f"   Beschreibung: {var_info['description']}")
                print(f"   Datentyp: {var_info['data_type']}")
                print(f"   Quelle: {var_info['source']}")
                
                if 'expected_range' in var_info:
                    print(f"   Erwarteter Bereich: {var_info['expected_range']}")
                if 'categories' in var_info:
                    print(f"   Kategorien: {var_info['categories']}")
                if 'threshold' in var_info:
                    print(f"   Schwellenwert: {var_info['threshold']}")
                if 'missing_values' in var_info:
                    print(f"   Fehlende Werte: {var_info['missing_values']}")
                if 'notes' in var_info:
                    print(f"   Notizen: {var_info['notes']}")
        
        print("\n" + "=" * 80)
        print("ENDE DER VARIABLEN-ÜBERSICHT")
        print("=" * 80) 