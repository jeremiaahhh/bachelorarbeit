"""
Demonstrationsskript für Training vs Test Vergleich
Zeigt die Schätzergebnisse für Trainings- und Test-Samples im Vergleich.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from config import Config
from evaluation_module import EvaluationModule
from input_module import InputModule

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """
    Erstellt Beispieldaten für die Demonstration
    
    Returns:
        DataFrame mit Beispieldaten
    """
    logger.info("Erstelle Beispieldaten für Training vs Test Vergleich")
    
    # Erstelle Daten für Training (2003-2019) und Test (2020-2023)
    train_start = datetime(2003, 1, 1)
    train_end = datetime(2019, 12, 31)
    test_start = datetime(2020, 1, 1)
    test_end = datetime(2023, 12, 31)
    
    # Trainingsdaten
    train_dates = pd.date_range(start=train_start, end=train_end, freq='D')
    test_dates = pd.date_range(start=test_start, end=test_end, freq='D')
    
    data = []
    
    # Trainingsdaten erstellen
    for date in train_dates:
        for indicator in ['GDP', 'CPIAUCSL', 'UNRATE', 'GS10']:
            # Simuliere realistische Werte
            if indicator == 'GDP':
                base_value = 20000 + np.random.normal(0, 500)
            elif indicator == 'CPIAUCSL':
                base_value = 250 + np.random.normal(0, 5)
            elif indicator == 'UNRATE':
                base_value = 5 + np.random.normal(0, 1)
            else:  # GS10
                base_value = 3 + np.random.normal(0, 0.5)
            
            # Standardisierte Werte
            standardized_value = np.random.normal(0, 1)
            
            # Kursänderungen (realistischer für Trainingsdaten)
            daily_change = np.random.normal(0, 0.3)
            weekly_change = np.random.normal(0, 1.0)
            monthly_change = np.random.normal(0, 2.0)
            yearly_change = np.random.normal(0, 10.0)
            
            # Richtungen
            daily_direction = 1 if daily_change > Config.MOVEMENT_THRESHOLD else 0
            weekly_direction = 1 if weekly_change > Config.MOVEMENT_THRESHOLD else 0
            monthly_direction = 1 if monthly_change > Config.MOVEMENT_THRESHOLD else 0
            yearly_direction = 1 if yearly_change > Config.MOVEMENT_THRESHOLD else 0
            
            data.append({
                'publication_date': date,
                'indicator': indicator,
                'indicator_value': base_value,
                'indicator_value_standardized': standardized_value,
                'daily_rate_change': daily_change,
                'weekly_rate_change': weekly_change,
                'monthly_rate_change': monthly_change,
                'yearly_rate_change': yearly_change,
                'daily_direction': daily_direction,
                'weekly_direction': weekly_direction,
                'monthly_direction': monthly_direction,
                'yearly_direction': yearly_direction
            })
    
    # Testdaten erstellen (mit etwas anderen Mustern)
    for date in test_dates:
        for indicator in ['GDP', 'CPIAUCSL', 'UNRATE', 'GS10']:
            # Simuliere realistische Werte (leicht andere Verteilung für Test)
            if indicator == 'GDP':
                base_value = 22000 + np.random.normal(0, 600)  # Höhere Werte
            elif indicator == 'CPIAUCSL':
                base_value = 270 + np.random.normal(0, 6)  # Höhere Inflation
            elif indicator == 'UNRATE':
                base_value = 6 + np.random.normal(0, 1.2)  # Höhere Arbeitslosigkeit
            else:  # GS10
                base_value = 2 + np.random.normal(0, 0.8)  # Niedrigere Zinsen
            
            # Standardisierte Werte
            standardized_value = np.random.normal(0, 1.1)  # Leicht andere Verteilung
            
            # Kursänderungen (realistischer für Testdaten - mehr Volatilität)
            daily_change = np.random.normal(0, 0.4)
            weekly_change = np.random.normal(0, 1.5)
            monthly_change = np.random.normal(0, 3.5)
            yearly_change = np.random.normal(0, 15.0)
            
            # Richtungen
            daily_direction = 1 if daily_change > Config.MOVEMENT_THRESHOLD else 0
            weekly_direction = 1 if weekly_change > Config.MOVEMENT_THRESHOLD else 0
            monthly_direction = 1 if monthly_change > Config.MOVEMENT_THRESHOLD else 0
            yearly_direction = 1 if yearly_change > Config.MOVEMENT_THRESHOLD else 0
            
            data.append({
                'publication_date': date,
                'indicator': indicator,
                'indicator_value': base_value,
                'indicator_value_standardized': standardized_value,
                'daily_rate_change': daily_change,
                'weekly_rate_change': weekly_change,
                'monthly_rate_change': monthly_change,
                'yearly_rate_change': yearly_change,
                'daily_direction': daily_direction,
                'weekly_direction': weekly_direction,
                'monthly_direction': monthly_direction,
                'yearly_direction': yearly_direction
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Beispieldaten erstellt: {len(df)} Zeilen")
    logger.info(f"Trainingsdaten: {len(df[df['publication_date'] < test_start])} Zeilen")
    logger.info(f"Testdaten: {len(df[df['publication_date'] >= test_start])} Zeilen")
    
    return df

def demonstrate_training_test_comparison():
    """
    Demonstriert den Training vs Test Vergleich
    """
    print("=" * 80)
    print("DEMONSTRATION: TRAINING VS TEST VERGLEICH")
    print("=" * 80)
    
    # 1. Erstelle Beispieldaten
    print("\n📊 ERSTELLE BEISPIELDATEN")
    print("-" * 60)
    sample_data = create_sample_data()
    
    # 2. Trainiere Modelle
    print("\nTRAINIERE MODELLE")
    print("-" * 60)
    
    evaluation_module = EvaluationModule()
    evaluation_results = evaluation_module.train_models(sample_data)
    
    if not evaluation_results:
        print("❌ Modelltraining fehlgeschlagen")
        return
    
    print(f"✅ {len(evaluation_results)} Modelle erfolgreich trainiert")
    
    # 3. Zeige detaillierte Ergebnisse
    print("\n📈 DETAILLIERTE ERGEBNISSE")
    print("-" * 60)
    
    for horizon, result in evaluation_results.items():
        print(f"\n🔹 Horizont: {horizon}")
        print(f"   Features: {result['n_features']}")
        print(f"   Trainings-Samples: {result['n_train_samples']:,}")
        print(f"   Test-Samples: {result['n_test_samples']:,}")
        
        # Trainingsdaten-Info
        train_info = result['train_data_info']
        print(f"   Trainingszeitraum: {train_info['start_date'].strftime('%Y-%m-%d')} bis {train_info['end_date'].strftime('%Y-%m-%d')}")
        
        # Testdaten-Info
        test_info = result['test_data_info']
        print(f"   Testzeitraum: {test_info['start_date'].strftime('%Y-%m-%d')} bis {test_info['end_date'].strftime('%Y-%m-%d')}")
        
        # Trainings-Metriken
        train_metrics = result['train_metrics']
        print(f"\n   📊 TRAINING:")
        print(f"      Accuracy:  {train_metrics['accuracy']:.4f}")
        print(f"      Precision: {train_metrics['precision']:.4f}")
        print(f"      Recall:    {train_metrics['recall']:.4f}")
        print(f"      F1-Score:  {train_metrics['f1_score']:.4f}")
        print(f"      ROC-AUC:   {train_metrics['roc_auc']:.4f}")
        
        # Test-Metriken
        test_metrics = result['test_metrics']
        print(f"\n   🧪 TEST:")
        print(f"      Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"      Precision: {test_metrics['precision']:.4f}")
        print(f"      Recall:    {test_metrics['recall']:.4f}")
        print(f"      F1-Score:  {test_metrics['f1_score']:.4f}")
        print(f"      ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        
        # Vergleich
        print(f"\n   ⚖️  VERGLEICH:")
        accuracy_diff = test_metrics['accuracy'] - train_metrics['accuracy']
        precision_diff = test_metrics['precision'] - train_metrics['precision']
        recall_diff = test_metrics['recall'] - train_metrics['recall']
        f1_diff = test_metrics['f1_score'] - train_metrics['f1_score']
        roc_auc_diff = test_metrics['roc_auc'] - train_metrics['roc_auc']
        
        print(f"      Accuracy-Differenz:  {accuracy_diff:+.4f}")
        print(f"      Precision-Differenz: {precision_diff:+.4f}")
        print(f"      Recall-Differenz:    {recall_diff:+.4f}")
        print(f"      F1-Differenz:        {f1_diff:+.4f}")
        print(f"      ROC-AUC-Differenz:   {roc_auc_diff:+.4f}")
        
        # Overfitting-Analyse
        if accuracy_diff < -0.05:
            overfitting_status = "⚠️  OVERFITTING (Test deutlich schlechter)"
        elif accuracy_diff < 0:
            overfitting_status = "⚠️  Leichtes Overfitting"
        elif accuracy_diff > 0.05:
            overfitting_status = "✅ Unterfitting (Test besser)"
        else:
            overfitting_status = "✅ Gute Generalisierung"
        
        print(f"      Status: {overfitting_status}")
    
    # 4. Zeige Zusammenfassungstabellen
    print("\n📋 ZUSAMMENFASSUNGSTABELLEN")
    print("-" * 60)
    
    # Detaillierte Zusammenfassung
    results_summary = evaluation_module.get_results_summary()
    print("\n📊 DETAILLIERTE ERGEBNISSE:")
    print(results_summary.to_string(index=False))
    
    # Vergleichszusammenfassung
    comparison_summary = evaluation_module.get_comparison_summary()
    print("\n⚖️  VERGLEICHSZUSAMMENFASSUNG:")
    print(comparison_summary.to_string(index=False))
    
    # 5. Overfitting-Analyse
    print("\n🔍 OVERFITTING-ANALYSE")
    print("-" * 60)
    
    overfitting_count = 0
    good_fit_count = 0
    
    for _, row in comparison_summary.iterrows():
        horizon = row['horizon']
        accuracy_diff = row['accuracy_diff']
        overfitting_indicator = row['overfitting_indicator']
        
        print(f"\n🔹 {horizon}:")
        print(f"   Accuracy-Differenz: {accuracy_diff:+.4f}")
        print(f"   Status: {overfitting_indicator}")
        
        if overfitting_indicator == 'Overfitting':
            overfitting_count += 1
            print(f"   ⚠️  Warnung: Modell zeigt Overfitting")
        else:
            good_fit_count += 1
            print(f"   ✅ Modell zeigt gute Generalisierung")
    
    print(f"\n📊 GESAMTÜBERSICHT:")
    print(f"   Modelle mit Overfitting: {overfitting_count}")
    print(f"   Modelle mit guter Generalisierung: {good_fit_count}")
    print(f"   Gesamtanzahl Modelle: {len(comparison_summary)}")
    
    # 6. Empfehlungen
    print("\n💡 EMPFEHLUNGEN")
    print("-" * 60)
    
    # Beste Modelle (basierend auf Test-Accuracy)
    best_models = comparison_summary.nlargest(2, 'test_accuracy')
    print("🏆 Beste Modelle (basierend auf Test-Accuracy):")
    for _, row in best_models.iterrows():
        print(f"   • {row['horizon']}: Test-Accuracy = {row['test_accuracy']:.4f}")
    
    # Modelle ohne Overfitting
    good_models = comparison_summary[comparison_summary['overfitting_indicator'] == 'Good Fit']
    if not good_models.empty:
        print("\n✅ Modelle ohne Overfitting:")
        for _, row in good_models.iterrows():
            print(f"   • {row['horizon']}: Accuracy-Diff = {row['accuracy_diff']:+.4f}")
    else:
        print("\n⚠️  Alle Modelle zeigen Anzeichen von Overfitting")
    
    # 7. Speichere Ergebnisse
    print(f"\n💾 SPEICHERE ERGEBNISSE")
    print("-" * 60)
    
    # Erstelle Verzeichnis falls nicht vorhanden
    import os
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Speichere Ergebnisse
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detaillierte Ergebnisse
    results_path = f"{Config.RESULTS_DIR}/training_test_comparison_demo_{timestamp}.csv"
    results_summary.to_csv(results_path, index=False)
    print(f"   Detaillierte Ergebnisse: {results_path}")
    
    # Vergleichszusammenfassung
    comparison_path = f"{Config.RESULTS_DIR}/training_test_comparison_summary_{timestamp}.csv"
    comparison_summary.to_csv(comparison_path, index=False)
    print(f"   Vergleichszusammenfassung: {comparison_path}")
    
    # 8. Erstelle Bericht
    report_path = f"{Config.RESULTS_DIR}/training_test_comparison_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TRAINING VS TEST VERGLEICH - DEMONSTRATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Anzahl Modelle: {len(evaluation_results)}\n")
        f.write(f"Trainingszeitraum: {Config.TRAINING_START} bis {Config.TRAINING_END}\n")
        f.write(f"Testzeitraum: {Config.TEST_START} bis {Config.TEST_END}\n\n")
        
        f.write("DETAILLIERTE ERGEBNISSE:\n")
        f.write("-" * 40 + "\n")
        f.write(results_summary.to_string())
        f.write("\n\n")
        
        f.write("VERGLEICHSZUSAMMENFASSUNG:\n")
        f.write("-" * 40 + "\n")
        f.write(comparison_summary.to_string())
        f.write("\n\n")
        
        f.write("OVERFITTING-ANALYSE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Modelle mit Overfitting: {overfitting_count}\n")
        f.write(f"Modelle mit guter Generalisierung: {good_fit_count}\n")
        f.write(f"Gesamtanzahl Modelle: {len(comparison_summary)}\n\n")
        
        f.write("EMPFOHLUNGEN:\n")
        f.write("-" * 40 + "\n")
        f.write("Beste Modelle (basierend auf Test-Accuracy):\n")
        for _, row in best_models.iterrows():
            f.write(f"  • {row['horizon']}: Test-Accuracy = {row['test_accuracy']:.4f}\n")
        
        if not good_models.empty:
            f.write("\nModelle ohne Overfitting:\n")
            for _, row in good_models.iterrows():
                f.write(f"  • {row['horizon']}: Accuracy-Diff = {row['accuracy_diff']:+.4f}\n")
    
    print(f"   Bericht: {report_path}")
    
    print(f"\n✅ DEMONSTRATION ABGESCHLOSSEN")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_training_test_comparison()
