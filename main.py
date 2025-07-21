"""
Hauptskript für das Devisenhandel-Vorhersagesystem
Orchestriert die gesamte Systemarchitektur und bietet einen vollständigen Workflow.
"""
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
import warnings
from typing import Dict

from config import Config
from input_module import InputModule
from evaluation_module import EvaluationModule
from output_module import OutputModule

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_prediction_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Warnungen unterdrücken
warnings.filterwarnings('ignore')

class ForexPredictionSystem:
    """Hauptklasse des Devisenhandel-Vorhersagesystems"""
    
    def __init__(self):
        """Initialisiert das System"""
        logger.info("Initialisiere Devisenhandel-Vorhersagesystem")
        
        # Erstelle Verzeichnisse
        Config.create_directories()
        
        # Initialisiere Module
        self.input_module = InputModule()
        self.evaluation_module = EvaluationModule()
        self.output_module = OutputModule()
        
        logger.info("System initialisiert")
    
    def run_complete_workflow(self, start_date: str = None, end_date: str = None,
                            load_existing_data: bool = False, 
                            data_filename: str = None) -> Dict:
        """
        Führt den vollständigen Workflow aus
        
        Args:
            start_date: Startdatum (optional, wenn load_existing_data=True)
            end_date: Enddatum (optional, wenn load_existing_data=True)
            load_existing_data: Ob existierende Daten geladen werden sollen
            data_filename: Dateiname für existierende Daten
            
        Returns:
            Dictionary mit allen Ergebnissen
        """
        logger.info("Starte vollständigen Workflow")
        
        # 1. Datenverarbeitung
        if load_existing_data and data_filename:
            logger.info(f"Lade existierende Daten: {data_filename}")
            event_matrix = self.input_module.load_data(data_filename)
        else:
            if not start_date or not end_date:
                start_date = Config.TRAINING_START
                end_date = Config.TEST_END
            
            logger.info(f"Verarbeite Daten von {start_date} bis {end_date}")
            event_matrix = self.input_module.process_data(start_date, end_date)
            
            # Speichere verarbeitete Daten
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = f"event_matrix_{timestamp}.csv"
            self.input_module.save_data(event_matrix, data_filename)
        
        if event_matrix.empty:
            logger.error("Keine Daten verfügbar")
            return {}
        
        # 2. Modelltraining
        logger.info("Starte Modelltraining")
        evaluation_results = self.evaluation_module.train_models(event_matrix)
        
        if not evaluation_results:
            logger.error("Modelltraining fehlgeschlagen")
            return {}
        
        # 3. Vorhersagen
        logger.info("Erstelle Vorhersagen")
        predictions_df = self.evaluation_module.predict(event_matrix)
        
        # 4. Ergebnisanalyse
        logger.info("Analysiere Ergebnisse")
        complete_results = self.output_module.process_and_analyze(
            predictions_df, event_matrix, evaluation_results
        )
        
        # 4.5. Erstelle umfassende Variablen-Zusammenfassung
        logger.info("Erstelle umfassende Variablen-Zusammenfassung")
        variable_summary = self.output_module.processor.create_comprehensive_variable_summary(event_matrix)
        complete_results['comprehensive_variable_summary'] = variable_summary
        
        # 5. Visualisierungen
        logger.info("Erstelle Visualisierungen")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_paths = self.output_module.create_visualizations(
            complete_results['processed_results'], 
            complete_results['evaluation_results'],
            save_plots=True
        )
        
        # 6. Speichere Ergebnisse
        logger.info("Speichere Ergebnisse")
        self.output_module.save_results(complete_results, f"results_{timestamp}")
        
        # 7. Generiere Bericht
        logger.info("Generiere Bericht")
        report = self.output_module.generate_report(complete_results, f"report_{timestamp}")
        
        # 8. Speichere Modelle
        logger.info("Speichere trainierte Modelle")
        self.evaluation_module.save_models(f"models_{timestamp}")
        
        # Zusammenfassung
        workflow_results = {
            'event_matrix': event_matrix,
            'predictions': predictions_df,
            'complete_results': complete_results,
            'plot_paths': plot_paths,
            'report': report,
            'data_filename': data_filename,
            'timestamp': timestamp
        }
        
        logger.info("Workflow abgeschlossen")
        return workflow_results
    
    def run_training_only(self, start_date: str = None, end_date: str = None,
                         load_existing_data: bool = False, 
                         data_filename: str = None) -> Dict:
        """
        Führt nur das Modelltraining aus
        
        Args:
            start_date: Startdatum
            end_date: Enddatum
            load_existing_data: Ob existierende Daten geladen werden sollen
            data_filename: Dateiname für existierende Daten
            
        Returns:
            Dictionary mit Trainingsergebnissen
        """
        logger.info("Starte Modelltraining")
        
        # Daten laden/verarbeiten
        if load_existing_data and data_filename:
            event_matrix = self.input_module.load_data(data_filename)
        else:
            if not start_date or not end_date:
                start_date = Config.TRAINING_START
                end_date = Config.TEST_END
            
            event_matrix = self.input_module.process_data(start_date, end_date)
        
        if event_matrix.empty:
            logger.error("Keine Daten verfügbar")
            return {}
        
        # Modelltraining
        evaluation_results = self.evaluation_module.train_models(event_matrix)
        
        # Ergebnisse speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evaluation_module.save_models(f"models_{timestamp}")
        
        return {
            'evaluation_results': evaluation_results,
            'timestamp': timestamp
        }
    
    def run_prediction_only(self, event_matrix: pd.DataFrame, 
                          model_prefix: str) -> pd.DataFrame:
        """
        Führt nur Vorhersagen aus (mit trainierten Modellen)
        
        Args:
            event_matrix: Ereignismatrix
            model_prefix: Präfix der trainierten Modelle
            
        Returns:
            DataFrame mit Vorhersagen
        """
        logger.info("Starte Vorhersagen mit trainierten Modellen")
        
        # Modelle laden
        self.evaluation_module.load_models(model_prefix)
        
        # Vorhersagen
        predictions_df = self.evaluation_module.predict(event_matrix)
        
        return predictions_df
    
    def run_analysis_only(self, predictions_df: pd.DataFrame, 
                         event_matrix: pd.DataFrame,
                         evaluation_results: Dict) -> Dict:
        """
        Führt nur die Ergebnisanalyse aus
        
        Args:
            predictions_df: Vorhersagen
            event_matrix: Ereignismatrix
            evaluation_results: Evaluationsergebnisse
            
        Returns:
            Dictionary mit Analyseergebnissen
        """
        logger.info("Starte Ergebnisanalyse")
        
        # Ergebnisanalyse
        complete_results = self.output_module.process_and_analyze(
            predictions_df, event_matrix, evaluation_results
        )
        
        # Visualisierungen
        plot_paths = self.output_module.create_visualizations(
            complete_results['processed_results'], 
            complete_results['evaluation_results'],
            save_plots=True
        )
        
        # Bericht generieren
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = self.output_module.generate_report(complete_results, f"report_{timestamp}")
        
        return {
            'complete_results': complete_results,
            'plot_paths': plot_paths,
            'report': report
        }
    
    def get_system_status(self) -> Dict:
        """
        Gibt den Systemstatus zurück
        
        Returns:
            Dictionary mit Systemstatus
        """
        status = {
            'system_initialized': True,
            'modules_loaded': {
                'input_module': self.input_module is not None,
                'evaluation_module': self.evaluation_module is not None,
                'output_module': self.output_module is not None
            },
            'models_trained': len(self.evaluation_module.models) > 0,
            'available_models': list(self.evaluation_module.models.keys()),
            'config': {
                'training_start': Config.TRAINING_START,
                'training_end': Config.TRAINING_END,
                'test_start': Config.TEST_START,
                'test_end': Config.TEST_END,
                'forecast_horizons': Config.FORECAST_HORIZONS,
                'fred_indicators': list(Config.FRED_INDICATORS.keys())
            }
        }
        
        return status

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='Devisenhandel-Vorhersagesystem')
    parser.add_argument('--mode', choices=['complete', 'training', 'prediction', 'analysis', 'status'],
                       default='complete', help='Ausführungsmodus')
    parser.add_argument('--start_date', type=str, help='Startdatum (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='Enddatum (YYYY-MM-DD)')
    parser.add_argument('--load_data', type=str, help='Dateiname für existierende Daten')
    parser.add_argument('--model_prefix', type=str, help='Präfix für trainierte Modelle')
    
    args = parser.parse_args()
    
    # System initialisieren
    system = ForexPredictionSystem()
    
    try:
        if args.mode == 'complete':
            # Vollständiger Workflow
            results = system.run_complete_workflow(
                start_date=args.start_date,
                end_date=args.end_date,
                load_existing_data=args.load_data is not None,
                data_filename=args.load_data
            )
            
            if results:
                print("Vollständiger Workflow erfolgreich abgeschlossen")
                print(f"Bericht: {results['report'][:500]}...")
            else:
                print("Workflow fehlgeschlagen")
        
        elif args.mode == 'training':
            # Nur Training
            results = system.run_training_only(
                start_date=args.start_date,
                end_date=args.end_date,
                load_existing_data=args.load_data is not None,
                data_filename=args.load_data
            )
            
            if results:
                print("Modelltraining erfolgreich abgeschlossen")
                print(f"Modelle gespeichert mit Präfix: models_{results['timestamp']}")
            else:
                print("Training fehlgeschlagen")
        
        elif args.mode == 'prediction':
            # Nur Vorhersagen
            if not args.load_data or not args.model_prefix:
                print("Für Vorhersagen müssen --load_data und --model_prefix angegeben werden")
                return
            
            # Daten laden
            event_matrix = system.input_module.load_data(args.load_data)
            predictions = system.run_prediction_only(event_matrix, args.model_prefix)
            
            if not predictions.empty:
                print("Vorhersagen erfolgreich erstellt")
                print(f"Anzahl Vorhersagen: {len(predictions)}")
            else:
                print("Vorhersagen fehlgeschlagen")
        
        elif args.mode == 'analysis':
            # Nur Analyse
            print("Analyse-Modus erfordert manuelle Eingabe der Daten")
            print("Verwenden Sie die Python-API für detaillierte Analysen")
        
        elif args.mode == 'status':
            # Systemstatus
            status = system.get_system_status()
            print("Systemstatus:")
            for key, value in status.items():
                print(f"  {key}: {value}")
    
    except Exception as e:
        logger.error(f"Fehler im Hauptprogramm: {e}")
        print(f"Fehler: {e}")

if __name__ == "__main__":
    main() 