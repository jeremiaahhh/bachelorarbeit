#!/usr/bin/env python3
"""
Demonstrationsskript für Training vs Test Visualisierungen
Erstellt Grafiken und Übersichten als Bilder für die neuen Trainings- vs Test-Vergleiche
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from output_module import OutputModule
from config import Config

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """
    Erstellt Beispieldaten für die Visualisierungsdemonstration
    """
    logger.info("Erstelle Beispieldaten für Visualisierungsdemonstration")
    
    # Erstelle verschiedene Horizonte
    horizons = ['daily', 'weekly', 'monthly', 'yearly']
    
    # Erstelle Beispieldaten für Training und Test
    sample_data = []
    
    for horizon in horizons:
        # Training-Daten (2003-2019)
        train_start = datetime(2003, 1, 1)
        train_end = datetime(2019, 12, 31)
        train_samples = np.random.randint(4000, 6000)  # Verschiedene Sample-Größen
        
        # Test-Daten (2020-2025)
        test_start = datetime(2020, 1, 1)
        test_end = datetime(2025, 12, 31)
        test_samples = np.random.randint(1000, 2000)
        
        # Training-Metriken (realistische Werte)
        train_accuracy = np.random.uniform(0.45, 0.55)
        train_precision = np.random.uniform(0.44, 0.56)
        train_recall = np.random.uniform(0.43, 0.57)
        train_f1_score = np.random.uniform(0.44, 0.56)
        train_roc_auc = np.random.uniform(0.48, 0.52)
        
        # Test-Metriken (leicht unterschiedlich für Realismus)
        test_accuracy = train_accuracy + np.random.uniform(-0.08, 0.08)
        test_precision = train_precision + np.random.uniform(-0.08, 0.08)
        test_recall = train_recall + np.random.uniform(-0.08, 0.08)
        test_f1_score = train_f1_score + np.random.uniform(-0.08, 0.08)
        test_roc_auc = train_roc_auc + np.random.uniform(-0.08, 0.08)
        
        # Training-Daten hinzufügen
        sample_data.append({
            'horizon': horizon,
            'sample_type': 'Training',
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1_score': train_f1_score,
            'roc_auc': train_roc_auc,
            'n_samples': train_samples,
            'n_features': 25,
            'start_date': train_start,
            'end_date': train_end
        })
        
        # Test-Daten hinzufügen
        sample_data.append({
            'horizon': horizon,
            'sample_type': 'Test',
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1_score,
            'roc_auc': test_roc_auc,
            'n_samples': test_samples,
            'n_features': 25,
            'start_date': test_start,
            'end_date': test_end
        })
    
    return pd.DataFrame(sample_data)

def create_comparison_data(results_summary):
    """
    Erstellt Vergleichsdaten aus den Ergebnissen
    """
    comparison_data = []
    
    for horizon in results_summary['horizon'].unique():
        train_data = results_summary[(results_summary['horizon'] == horizon) & 
                                   (results_summary['sample_type'] == 'Training')]
        test_data = results_summary[(results_summary['horizon'] == horizon) & 
                                  (results_summary['sample_type'] == 'Test')]
        
        if not train_data.empty and not test_data.empty:
            train_row = train_data.iloc[0]
            test_row = test_data.iloc[0]
            
            comparison_data.append({
                'horizon': horizon,
                'train_accuracy': train_row['accuracy'],
                'test_accuracy': test_row['accuracy'],
                'accuracy_diff': test_row['accuracy'] - train_row['accuracy'],
                'train_precision': train_row['precision'],
                'test_precision': test_row['precision'],
                'precision_diff': test_row['precision'] - train_row['precision'],
                'train_recall': train_row['recall'],
                'test_recall': test_row['recall'],
                'recall_diff': test_row['recall'] - train_row['recall'],
                'train_f1_score': train_row['f1_score'],
                'test_f1_score': test_row['f1_score'],
                'f1_diff': test_row['f1_score'] - train_row['f1_score'],
                'train_roc_auc': train_row['roc_auc'],
                'test_roc_auc': test_row['roc_auc'],
                'roc_auc_diff': test_row['roc_auc'] - train_row['roc_auc'],
                'n_train_samples': train_row['n_samples'],
                'n_test_samples': test_row['n_samples'],
                'n_features': train_row['n_features'],
                'overfitting_indicator': 'Overfitting' if test_row['accuracy'] < train_row['accuracy'] - 0.05 else 'Good Fit'
            })
    
    return pd.DataFrame(comparison_data)

def create_sample_feature_importance_data():
    """
    Erstellt Beispieldaten für Feature-Importance Visualisierungen
    """
    logger.info("Erstelle Beispieldaten für Feature-Importance")
    
    # Erstelle verschiedene Horizonte
    horizons = ['daily', 'weekly', 'monthly', 'yearly']
    
    # Erstelle Beispielfeatures
    features = [
        'GDP_growth', 'CPI_yoy', 'PPI_yoy', 'interest_rate_spread',
        'unemployment_rate', 'consumer_confidence', 'GDP_lag_1', 'CPI_lag_1',
        'PPI_lag_1', 'interest_spread_lag_1', 'unemployment_lag_1', 'confidence_lag_1',
        'GDP_rolling_mean', 'CPI_rolling_mean', 'PPI_rolling_mean', 'spread_rolling_mean',
        'unemployment_rolling_mean', 'confidence_rolling_mean', 'year_dummy', 'month_dummy',
        'quarter_dummy', 'day_of_week_dummy', 'GDP_std', 'CPI_std', 'PPI_std', 'spread_std'
    ]
    
    sample_evaluation_results = {}
    
    for horizon in horizons:
        # Erstelle Training Feature-Importance
        train_importance_data = []
        for i, feature in enumerate(features):
            # Realistische Koeffizienten mit Variation
            base_coef = np.random.uniform(-0.5, 0.5)
            train_coef = base_coef + np.random.uniform(-0.1, 0.1)
            
            train_importance_data.append({
                'feature': feature,
                'coefficient': train_coef,
                'abs_coefficient': abs(train_coef)
            })
        
        train_importance_df = pd.DataFrame(train_importance_data)
        train_importance_df = train_importance_df.sort_values('abs_coefficient', ascending=False)
        
        # Erstelle Test Feature-Importance (leicht unterschiedlich)
        test_importance_data = []
        for i, feature in enumerate(features):
            base_coef = np.random.uniform(-0.5, 0.5)
            test_coef = base_coef + np.random.uniform(-0.1, 0.1)
            
            test_importance_data.append({
                'feature': feature,
                'coefficient': test_coef,
                'abs_coefficient': abs(test_coef)
            })
        
        test_importance_df = pd.DataFrame(test_importance_data)
        test_importance_df = test_importance_df.sort_values('abs_coefficient', ascending=False)
        
        # Erstelle Beispieldaten für Confusion Matrix
        train_confusion_matrix = np.array([
            [np.random.randint(800, 1200), np.random.randint(200, 400)],
            [np.random.randint(200, 400), np.random.randint(800, 1200)]
        ])
        
        test_confusion_matrix = np.array([
            [np.random.randint(200, 300), np.random.randint(50, 100)],
            [np.random.randint(50, 100), np.random.randint(200, 300)]
        ])
        
        # Berechne Metriken aus Confusion Matrix
        train_tp = train_confusion_matrix[1, 1]
        train_fp = train_confusion_matrix[0, 1]
        train_fn = train_confusion_matrix[1, 0]
        train_tn = train_confusion_matrix[0, 0]
        
        test_tp = test_confusion_matrix[1, 1]
        test_fp = test_confusion_matrix[0, 1]
        test_fn = test_confusion_matrix[1, 0]
        test_tn = test_confusion_matrix[0, 0]
        
        # Berechne Precision, Recall, F1-Score
        train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
        train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
        train_f1_score = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        
        test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
        test_recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
        test_f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
        
        sample_evaluation_results[horizon] = {
            'train_feature_importance': train_importance_df,
            'test_feature_importance': test_importance_df,
            'train_metrics': {
                'precision': train_precision,
                'recall': train_recall,
                'f1_score': train_f1_score,
                'confusion_matrix': train_confusion_matrix
            },
            'test_metrics': {
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1_score,
                'confusion_matrix': test_confusion_matrix
            }
        }
    
    return sample_evaluation_results

def demonstrate_training_test_visualizations():
    """
    Demonstriert die neuen Training vs Test Visualisierungen
    """
    logger.info("🚀 Starte Training vs Test Visualisierungsdemonstration")
    
    # 1. Erstelle Beispieldaten
    logger.info("📊 Erstelle Beispieldaten...")
    results_summary = create_sample_data()
    comparison_summary = create_comparison_data(results_summary)
    
    # 2. Initialisiere Output-Modul
    logger.info("🔧 Initialisiere Output-Modul...")
    output_module = OutputModule()
    
    # 3. Erstelle Visualisierungen
    logger.info("🎨 Erstelle neue Visualisierungen...")
    
    # Zeitstempel für Dateinamen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = f"training_test_visualization_demo_{timestamp}"
    
    # 1. Training vs Test Vergleich
    logger.info("📈 Erstelle Training vs Test Vergleich...")
    training_test_plot_path = f"{Config.PLOTS_DIR}/{filename_prefix}_training_test_comparison.png"
    output_module.visualizer.plot_training_test_comparison(
        results_summary, 
        save_path=training_test_plot_path
    )
    
    # 2. Overfitting-Analyse
    logger.info("🔍 Erstelle Overfitting-Analyse...")
    overfitting_plot_path = f"{Config.PLOTS_DIR}/{filename_prefix}_overfitting_analysis.png"
    output_module.visualizer.plot_overfitting_analysis(
        comparison_summary, 
        save_path=overfitting_plot_path
    )
    
    # 3. Metriken Heatmap
    logger.info("🔥 Erstelle Metriken Heatmap...")
    metrics_heatmap_path = f"{Config.PLOTS_DIR}/{filename_prefix}_metrics_heatmap.png"
    output_module.visualizer.plot_metrics_heatmap(
        results_summary, 
        save_path=metrics_heatmap_path
    )
    
    # 4. Feature-Importance Vergleich (mit Beispieldaten)
    logger.info("📊 Erstelle Feature-Importance Vergleich...")
    feature_importance_comparison_path = f"{Config.PLOTS_DIR}/{filename_prefix}_feature_importance_comparison.png"
    
    # Erstelle Beispieldaten für Feature-Importance
    sample_evaluation_results = create_sample_feature_importance_data()
    output_module.visualizer.plot_feature_importance_comparison(
        sample_evaluation_results, 
        save_path=feature_importance_comparison_path
    )
    
    # 5. Feature-Importance Heatmap
    logger.info("🔥 Erstelle Feature-Importance Heatmap...")
    feature_importance_heatmap_path = f"{Config.PLOTS_DIR}/{filename_prefix}_feature_importance_heatmap.png"
    output_module.visualizer.plot_feature_importance_heatmap(
        sample_evaluation_results, 
        save_path=feature_importance_heatmap_path
    )
    
    # 6. Training vs Test Confusion Matrizen
    logger.info("📊 Erstelle Training vs Test Confusion Matrizen...")
    confusion_matrices_path = f"{Config.PLOTS_DIR}/{filename_prefix}_confusion_matrices.png"
    output_module.visualizer.plot_confusion_matrices(
        sample_evaluation_results, 
        save_path=confusion_matrices_path
    )
    
    # 7. Confusion Matrix Metriken Vergleich
    logger.info("📈 Erstelle Confusion Matrix Metriken Vergleich...")
    confusion_metrics_comparison_path = f"{Config.PLOTS_DIR}/{filename_prefix}_confusion_metrics_comparison.png"
    output_module.visualizer.plot_confusion_matrix_metrics_comparison(
        sample_evaluation_results, 
        save_path=confusion_metrics_comparison_path
    )
    
    # 8. Signal-Analyse (Detaillierte Analyse der richtigen/falschen Signale)
    logger.info("📊 Erstelle Signal-Analyse...")
    signal_analysis_path = f"{Config.PLOTS_DIR}/{filename_prefix}_signal_analysis.png"
    output_module.visualizer.plot_signal_analysis(
        sample_evaluation_results, 
        save_path=signal_analysis_path
    )
    
    # 9. Signal-Zusammenfassungsvergleich
    logger.info("📈 Erstelle Signal-Zusammenfassungsvergleich...")
    signal_summary_comparison_path = f"{Config.PLOTS_DIR}/{filename_prefix}_signal_summary_comparison.png"
    output_module.visualizer.plot_signal_summary_comparison(
        sample_evaluation_results, 
        save_path=signal_summary_comparison_path
    )
    
    # 10. Zeige Zusammenfassung
    logger.info("📋 Zeige Zusammenfassung der erstellten Grafiken:")
    logger.info(f"   📊 Training vs Test Vergleich: {training_test_plot_path}")
    logger.info(f"   🔍 Overfitting-Analyse: {overfitting_plot_path}")
    logger.info(f"   🔥 Metriken Heatmap: {metrics_heatmap_path}")
    logger.info(f"   📊 Feature-Importance Vergleich: {feature_importance_comparison_path}")
    logger.info(f"   🔥 Feature-Importance Heatmap: {feature_importance_heatmap_path}")
    logger.info(f"   📊 Training vs Test Confusion Matrizen: {confusion_matrices_path}")
    logger.info(f"   📈 Confusion Matrix Metriken Vergleich: {confusion_metrics_comparison_path}")
    logger.info(f"   📊 Signal-Analyse: {signal_analysis_path}")
    logger.info(f"   📈 Signal-Zusammenfassungsvergleich: {signal_summary_comparison_path}")
    
    # 5. Zeige Datenübersicht
    logger.info("📊 Datenübersicht:")
    logger.info(f"   Anzahl Horizonte: {len(results_summary['horizon'].unique())}")
    logger.info(f"   Sample-Typen: {list(results_summary['sample_type'].unique())}")
    logger.info(f"   Metriken: {list(results_summary.columns[2:7])}")  # accuracy bis roc_auc
    
    # 6. Zeige Overfitting-Status
    logger.info("🎯 Overfitting-Status:")
    overfitting_counts = comparison_summary['overfitting_indicator'].value_counts()
    for status, count in overfitting_counts.items():
        logger.info(f"   {status}: {count} Horizonte")
    
    logger.info("✅ Training vs Test Visualisierungsdemonstration abgeschlossen!")
    
    return {
        'results_summary': results_summary,
        'comparison_summary': comparison_summary,
        'plot_paths': {
            'training_test_comparison': training_test_plot_path,
            'overfitting_analysis': overfitting_plot_path,
            'metrics_heatmap': metrics_heatmap_path,
            'feature_importance_comparison': feature_importance_comparison_path,
            'feature_importance_heatmap': feature_importance_heatmap_path,
            'confusion_matrices': confusion_matrices_path,
            'confusion_metrics_comparison': confusion_metrics_comparison_path,
            'signal_analysis': signal_analysis_path,
            'signal_summary_comparison': signal_summary_comparison_path
        }
    }

if __name__ == "__main__":
    try:
        results = demonstrate_training_test_visualizations()
        print("\n🎉 Demonstration erfolgreich abgeschlossen!")
        print("📁 Überprüfen Sie das 'plots/' Verzeichnis für die neuen Grafiken.")
        
    except Exception as e:
        logger.error(f"❌ Fehler bei der Demonstration: {e}")
        raise
