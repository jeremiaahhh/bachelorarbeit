"""
Demonstrationsskript für die umfassende Variablen-Zusammenfassung
Zeigt die neuen Summary-Statistiken für alle verwendeten Variablen.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from config import Config
from output_module import ResultProcessor

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """
    Erstellt Beispieldaten für die Demonstration
    
    Returns:
        DataFrame mit Beispieldaten
    """
    logger.info("Erstelle Beispieldaten für Demonstration")
    
    # Zeitraum
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Erstelle Beispieldaten
    data = []
    
    for date in dates:
        # Simuliere verschiedene Indikatoren
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
            
            # Kursänderungen
            daily_change = np.random.normal(0, 0.5)
            weekly_change = np.random.normal(0, 1.5)
            monthly_change = np.random.normal(0, 3.0)
            yearly_change = np.random.normal(0, 15.0)
            
            # Richtungen
            daily_direction = 1 if daily_change > Config.MOVEMENT_THRESHOLD else 0
            weekly_direction = 1 if weekly_change > Config.MOVEMENT_THRESHOLD else 0
            monthly_direction = 1 if monthly_change > Config.MOVEMENT_THRESHOLD else 0
            yearly_direction = 1 if yearly_change > Config.MOVEMENT_THRESHOLD else 0
            
            # Vorhersagen
            daily_probability = np.random.uniform(0, 1)
            daily_prediction = 1 if daily_probability > Config.CLASSIFICATION_THRESHOLD else 0
            
            # Intensität
            if daily_probability < 0.6:
                daily_intensity = 'weak'
            elif daily_probability < 0.75:
                daily_intensity = 'moderate'
            else:
                daily_intensity = 'strong'
            
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
                'yearly_direction': yearly_direction,
                'daily_prediction': daily_prediction,
                'daily_probability': daily_probability,
                'daily_intensity': daily_intensity,
                'daily_correct': daily_prediction == daily_direction
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Beispieldaten erstellt: {len(df)} Zeilen, {len(df.columns)} Spalten")
    return df

def demonstrate_variable_summary():
    """
    Demonstriert die umfassende Variablen-Zusammenfassung
    """
    print("=" * 80)
    print("DEMONSTRATION: UMFASSENDE VARIABLEN-ZUSAMMENFASSUNG")
    print("=" * 80)
    
    # 1. Zeige Variablen-Dokumentation
    print("\n📋 VARIABLEN-DOKUMENTATION AUS KONFIGURATION")
    print("-" * 60)
    Config.print_variable_summary()
    
    # 2. Erstelle Beispieldaten
    print("\n📊 ERSTELLE BEISPIELDATEN")
    print("-" * 60)
    sample_data = create_sample_data()
    
    # 3. Erstelle umfassende Zusammenfassung
    print("\n🔍 ERSTELLE UMFASSENDE VARIABLEN-ZUSAMMENFASSUNG")
    print("-" * 60)
    
    processor = ResultProcessor()
    comprehensive_summary = processor.create_comprehensive_variable_summary(sample_data)
    
    # 4. Zeige Ergebnisse
    print("\n📈 ERGEBNISSE DER UMFASSENDEN ANALYSE")
    print("-" * 60)
    
    # Datenübersicht
    data_overview = comprehensive_summary['data_overview']
    print(f"\n📊 DATENÜBERSICHT:")
    print(f"   Gesamtzeilen: {data_overview['total_rows']:,}")
    print(f"   Gesamtspalten: {data_overview['total_columns']}")
    print(f"   Speicherverbrauch: {data_overview['memory_usage_mb']:.2f} MB")
    print(f"   Zeitraum: {data_overview['date_range']['start']} bis {data_overview['date_range']['end']}")
    
    # Fehlende Daten
    missing_analysis = comprehensive_summary['missing_data_analysis']
    print(f"\n❌ FEHLENDE DATEN:")
    print(f"   Gesamtfehlende Zellen: {missing_analysis['total_missing_cells']:,}")
    print(f"   Gesamtvollständigkeit: {missing_analysis['overall_completeness']:.2f}%")
    print(f"   Variablen mit fehlenden Daten: {len(missing_analysis['variables_with_missing_data'])}")
    print(f"   Variablen ohne fehlende Daten: {len(missing_analysis['variables_without_missing_data'])}")
    
    # Zeitliche Analyse
    temporal_analysis = comprehensive_summary['temporal_analysis']
    print(f"\n⏰ ZEITLICHE ANALYSE:")
    print(f"   Zeitraum: {temporal_analysis['date_range']['duration_days']} Tage")
    print(f"   Fehlende Datumswerte: {temporal_analysis['data_gaps']['total_missing_dates']}")
    print(f"   Größte Datenlücke: {temporal_analysis['data_gaps']['largest_gap_days']} Tage")
    
    # Korrelationsanalyse
    correlation_analysis = comprehensive_summary['correlation_analysis']
    if 'highest_correlations' in correlation_analysis:
        print(f"\n🔗 STÄRKSTE KORRELATIONEN:")
        for i, corr in enumerate(correlation_analysis['highest_correlations'][:5]):
            print(f"   {i+1}. {corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']:.3f}")
    
    # Detaillierte Variablenanalyse
    print(f"\n📋 DETAILLIERTE VARIABLENANALYSE:")
    print("-" * 60)
    
    for category, variables in comprehensive_summary.items():
        if category not in ['data_overview', 'correlation_analysis', 'missing_data_analysis', 'temporal_analysis']:
            print(f"\n📊 {category.upper().replace('_', ' ')}:")
            
            for var_name, var_stats in variables.items():
                if 'basic_stats' in var_stats:
                    basic = var_stats['basic_stats']
                    print(f"   🔹 {var_name}:")
                    print(f"      Anzahl: {basic['count']:,}")
                    print(f"      Fehlende: {basic['missing_count']:,} ({basic['missing_percentage']:.1f}%)")
                    
                    if 'mean' in basic:
                        print(f"      Mittelwert: {basic['mean']:.3f}")
                        print(f"      Std: {basic['std']:.3f}")
                        print(f"      Min/Max: {basic['min']:.3f} / {basic['max']:.3f}")
                    
                    if 'quality_stats' in var_stats:
                        quality = var_stats['quality_stats']
                        print(f"      Qualitäts-Score: {quality['data_quality_score']:.1f}/100")
    
    # 5. Speichere Ergebnisse
    print(f"\n💾 SPEICHERE ERGEBNISSE")
    print("-" * 60)
    
    # Erstelle Verzeichnis falls nicht vorhanden
    import os
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Speichere als JSON
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"{Config.RESULTS_DIR}/comprehensive_variable_summary_demo_{timestamp}.json"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_summary, f, indent=2, default=str)
    
    print(f"   Umfassende Zusammenfassung gespeichert: {summary_path}")
    
    # 6. Erstelle Zusammenfassungsbericht
    report_path = f"{Config.RESULTS_DIR}/variable_summary_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("UMFASSENDE VARIABLEN-ZUSAMMENFASSUNG - DEMONSTRATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Datensatz: {data_overview['total_rows']:,} Zeilen, {data_overview['total_columns']} Spalten\n")
        f.write(f"Zeitraum: {data_overview['date_range']['start']} bis {data_overview['date_range']['end']}\n\n")
        
        f.write("DATENQUALITÄT:\n")
        f.write(f"- Gesamtvollständigkeit: {missing_analysis['overall_completeness']:.2f}%\n")
        f.write(f"- Fehlende Zellen: {missing_analysis['total_missing_cells']:,}\n")
        f.write(f"- Variablen mit fehlenden Daten: {len(missing_analysis['variables_with_missing_data'])}\n\n")
        
        f.write("ZEITLICHE ANALYSE:\n")
        f.write(f"- Zeitraum: {temporal_analysis['date_range']['duration_days']} Tage\n")
        f.write(f"- Fehlende Datumswerte: {temporal_analysis['data_gaps']['total_missing_dates']}\n")
        f.write(f"- Größte Datenlücke: {temporal_analysis['data_gaps']['largest_gap_days']} Tage\n\n")
        
        f.write("VARIABLENÜBERSICHT:\n")
        for category, variables in comprehensive_summary.items():
            if category not in ['data_overview', 'correlation_analysis', 'missing_data_analysis', 'temporal_analysis']:
                f.write(f"\n{category.upper()}:\n")
                for var_name, var_stats in variables.items():
                    if 'basic_stats' in var_stats:
                        basic = var_stats['basic_stats']
                        f.write(f"  - {var_name}: {basic['count']:,} Werte, {basic['missing_percentage']:.1f}% fehlend\n")
    
    print(f"   Bericht gespeichert: {report_path}")
    
    print(f"\n✅ DEMONSTRATION ABGESCHLOSSEN")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_variable_summary()
