"""
Beispielskript für die Verwendung des Devisenhandel-Vorhersagesystems
Demonstriert verschiedene Anwendungsfälle und Workflows.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from main import ForexPredictionSystem
from config import Config

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_complete_workflow():
    """Beispiel für den vollständigen Workflow"""
    print("=" * 60)
    print("BEISPIEL: Vollständiger Workflow")
    print("=" * 60)
    
    # System initialisieren
    system = ForexPredictionSystem()
    
    # Vollständigen Workflow ausführen
    results = system.run_complete_workflow(
        start_date='2015-01-01',  # Kürzerer Zeitraum für Demo
        end_date='2023-12-31'
    )
    
    if results:
        print("✅ Workflow erfolgreich abgeschlossen!")
        print(f"📊 Anzahl Ereignisse: {len(results['event_matrix'])}")
        print(f"📈 Anzahl Vorhersagen: {len(results['predictions'])}")
        print(f"📁 Daten gespeichert: {results['data_filename']}")
        print(f"🕒 Timestamp: {results['timestamp']}")
        
        # Zeige Zusammenfassung der Ergebnisse
        summary = results['complete_results']['results_summary']
        print("\n📋 Ergebnisübersicht:")
        print(summary.to_string(index=False))
        
        return results
    else:
        print("❌ Workflow fehlgeschlagen")
        return None

def example_training_only():
    """Beispiel für Modelltraining ohne vollständige Analyse"""
    print("\n" + "=" * 60)
    print("BEISPIEL: Nur Modelltraining")
    print("=" * 60)
    
    system = ForexPredictionSystem()
    
    # Nur Training ausführen
    results = system.run_training_only(
        start_date='2015-01-01',
        end_date='2020-12-31'
    )
    
    if results:
        print("✅ Modelltraining erfolgreich!")
        print(f"📁 Modelle gespeichert mit Präfix: models_{results['timestamp']}")
        
        # Zeige Trainingsergebnisse
        for horizon, result in results['evaluation_results'].items():
            metrics = result['test_metrics']
            print(f"\n📊 {horizon} Horizont:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return results
    else:
        print("❌ Training fehlgeschlagen")
        return None

def example_prediction_with_existing_models():
    """Beispiel für Vorhersagen mit trainierten Modellen"""
    print("\n" + "=" * 60)
    print("BEISPIEL: Vorhersagen mit existierenden Modellen")
    print("=" * 60)
    
    system = ForexPredictionSystem()
    
    # Beispiel: Lade existierende Daten (falls verfügbar)
    try:
        # Versuche, die neueste Ereignismatrix zu laden
        import os
        data_files = [f for f in os.listdir(Config.DATA_DIR) if f.startswith('event_matrix_')]
        if data_files:
            latest_file = max(data_files)
            print(f"📂 Lade existierende Daten: {latest_file}")
            
            event_matrix = system.input_module.load_data(latest_file)
            
            # Extrahiere Modell-Präfix aus Dateiname
            model_prefix = latest_file.replace('event_matrix_', 'models_').replace('.csv', '')
            
            # Mache Vorhersagen
            predictions = system.run_prediction_only(event_matrix, model_prefix)
            
            if not predictions.empty:
                print("✅ Vorhersagen erfolgreich erstellt!")
                print(f"📊 Anzahl Vorhersagen: {len(predictions)}")
                
                # Zeige Beispiel-Vorhersagen
                print("\n📋 Beispiel-Vorhersagen:")
                sample_cols = ['publication_date', 'indicator', 'daily_probability', 'daily_prediction', 'daily_intensity']
                available_cols = [col for col in sample_cols if col in predictions.columns]
                print(predictions[available_cols].head().to_string(index=False))
                
                return predictions
            else:
                print("❌ Vorhersagen fehlgeschlagen")
        else:
            print("ℹ️ Keine existierenden Daten gefunden")
            print("Führen Sie zuerst den vollständigen Workflow aus")
    
    except Exception as e:
        print(f"❌ Fehler beim Laden der Daten: {e}")
    
    return None

def example_system_status():
    """Beispiel für Systemstatus-Abfrage"""
    print("\n" + "=" * 60)
    print("BEISPIEL: Systemstatus")
    print("=" * 60)
    
    system = ForexPredictionSystem()
    status = system.get_system_status()
    
    print("📊 Systemstatus:")
    print(f"   System initialisiert: {status['system_initialized']}")
    print(f"   Module geladen: {status['modules_loaded']}")
    print(f"   Modelle trainiert: {status['models_trained']}")
    print(f"   Verfügbare Modelle: {status['available_models']}")
    
    print("\n⚙️ Konfiguration:")
    config = status['config']
    print(f"   Training: {config['training_start']} bis {config['training_end']}")
    print(f"   Test: {config['test_start']} bis {config['test_end']}")
    print(f"   Prognosehorizonte: {config['forecast_horizons']}")
    print(f"   FRED-Indikatoren: {len(config['fred_indicators'])} verfügbar")
    
    return status

def example_custom_analysis():
    """Beispiel für benutzerdefinierte Analysen"""
    print("\n" + "=" * 60)
    print("BEISPIEL: Benutzerdefinierte Analyse")
    print("=" * 60)
    
    system = ForexPredictionSystem()
    
    # Erstelle Beispiel-Daten für Demo
    print("🔧 Erstelle Beispiel-Daten für Demo...")
    
    # Simuliere eine kleine Ereignismatrix
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
    sample_data = []
    
    indicators = ['UNRATE', 'CPIAUCSL', 'PAYEMS']
    for date in dates:
        for indicator in indicators:
            sample_data.append({
                'publication_date': date,
                'indicator': indicator,
                'indicator_value': np.random.normal(100, 10),
                'indicator_value_standardized': np.random.normal(0, 1),
                'daily_rate_change': np.random.normal(0, 0.5),
                'daily_direction': np.random.choice([0, 1]),
                'weekly_rate_change': np.random.normal(0, 1.0),
                'weekly_direction': np.random.choice([0, 1]),
                'monthly_rate_change': np.random.normal(0, 2.0),
                'monthly_direction': np.random.choice([0, 1]),
                'yearly_rate_change': np.random.normal(0, 5.0),
                'yearly_direction': np.random.choice([0, 1])
            })
    
    event_matrix = pd.DataFrame(sample_data)
    
    print(f"📊 Beispiel-Daten erstellt: {len(event_matrix)} Ereignisse")
    
    # Analysiere die Daten
    print("\n📈 Analysiere Daten...")
    
    # Statistiken nach Indikator
    indicator_stats = event_matrix.groupby('indicator').agg({
        'daily_rate_change': ['mean', 'std'],
        'daily_direction': 'mean'
    }).round(4)
    
    print("\n📋 Statistiken nach Indikator:")
    print(indicator_stats)
    
    # Zeitliche Verteilung
    monthly_stats = event_matrix.groupby(event_matrix['publication_date'].dt.month).agg({
        'daily_rate_change': 'mean',
        'daily_direction': 'mean'
    }).round(4)
    
    print("\n📅 Monatliche Durchschnitte:")
    print(monthly_stats)
    
    return event_matrix

def example_feature_importance_analysis():
    """Beispiel für Feature-Importance-Analyse"""
    print("\n" + "=" * 60)
    print("BEISPIEL: Feature-Importance-Analyse")
    print("=" * 60)
    
    # Simuliere Feature-Importance-Daten
    features = [
        'UNRATE_standardized', 'CPIAUCSL_standardized', 'PAYEMS_standardized',
        'indicator_UNRATE', 'indicator_CPIAUCSL', 'indicator_PAYEMS',
        'year_2023', 'month_1', 'quarter_1'
    ]
    
    # Simuliere Koeffizienten
    coefficients = np.random.normal(0, 0.5, len(features))
    
    importance_df = pd.DataFrame({
        'feature': features,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("📊 Top 10 wichtigste Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Visualisiere Feature-Importance (ohne Anzeige für Kompatibilität)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'][:10], importance_df['abs_coefficient'][:10])
        plt.xlabel('Absolute Koeffizienten')
        plt.title('Feature Importance (Top 10)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()  # Schließe das Fenster ohne Anzeige
        print("📊 Feature-Importance-Plot gespeichert: feature_importance.png")
    except Exception as e:
        print(f"⚠️ Visualisierung übersprungen: {e}")
    
    return importance_df

def main():
    """Hauptfunktion für alle Beispiele"""
    print("🚀 DEVISENHANDEL-VORHERGESYSTEM - BEISPIELE")
    print("=" * 60)
    
    try:
        # 1. Systemstatus
        example_system_status()
        
        # 2. Vollständiger Workflow (kommentiert für Demo)
        print("\n" + "=" * 60)
        print("HINWEIS: Vollständiger Workflow erfordert FRED API-Schlüssel")
        print("Entkommentieren Sie die folgende Zeile nach Konfiguration:")
        print("# example_complete_workflow()")
        
        # Uncomment nach API-Schlüssel-Konfiguration:
        # example_complete_workflow()
        
        # 3. Nur Training (kommentiert für Demo)
        print("\n" + "=" * 60)
        print("HINWEIS: Modelltraining erfordert FRED API-Schlüssel")
        print("Entkommentieren Sie die folgende Zeile nach Konfiguration:")
        print("# example_training_only()")
        
        # Uncomment nach API-Schlüssel-Konfiguration:
        # example_training_only()
        
        # 4. Vorhersagen mit existierenden Modellen
        example_prediction_with_existing_models()
        
        # 5. Benutzerdefinierte Analyse
        example_custom_analysis()
        
        # 6. Feature-Importance-Analyse
        example_feature_importance_analysis()
        
        print("\n" + "=" * 60)
        print("✅ Alle Beispiele abgeschlossen!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Fehler in den Beispielen: {e}")
        logger.error(f"Fehler in den Beispielen: {e}")

if __name__ == "__main__":
    main() 