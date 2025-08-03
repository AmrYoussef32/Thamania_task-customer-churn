#!/usr/bin/env python3
"""
Model Retraining System

Handles the retraining of customer churn prediction models based on performance
and drift detection results.
"""

import os
import json
import pickle
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

# Import the main training script
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from customer_churn_prediction import CustomerChurnPredictor


class ModelRetrainer:
    """Handles model retraining based on performance and drift detection."""
    
    def __init__(self, config_path="../../config/retraining_config.json"):
        """Initialize the retrainer with configuration."""
        self.config = self._load_config(config_path)
        self.models_dir = self.config.get('models_dir', '../../models')
        self.backup_dir = self.config.get('backup_dir', '../../models/backup')
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def _load_config(self, config_path):
        """Load retraining configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration."""
        return {
            'f1_threshold': 0.7,
            'auc_threshold': 0.6,
            'precision_threshold': 0.5,
            'recall_threshold': 0.6,
            'drift_threshold': 0.3,
            'data_path': '../../data/customer_churn_mini.json',
            'models_dir': '../../models',
            'backup_dir': '../../models/backup',
            'logging': True,
            'min_improvement': 0.01
        }
    
    def _get_current_model_metrics(self):
        """Get performance metrics of the current model."""
        try:
            # Load current model and evaluate
            predictor = CustomerChurnPredictor()
            results = predictor.run_complete_pipeline()
            
            return {
                'f1_score': results.get('f1_score', 0.0),
                'auc_score': results.get('auc_score', 0.0),
                'precision': results.get('precision', 0.0),
                'recall': results.get('recall', 0.0)
            }
        except Exception as e:
            print(f"Error getting current model metrics: {e}")
            return None
    
    def _backup_current_model(self):
        """Backup the current model before retraining."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"model_backup_{timestamp}")
            
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Copy current model files
            current_files = [
                'churn_model_*.pkl',
                'preprocessing_*.pkl', 
                'feature_columns_*.json'
            ]
            
            import glob
            for pattern in current_files:
                for file_path in glob.glob(os.path.join(self.models_dir, pattern)):
                    filename = os.path.basename(file_path)
                    shutil.copy2(file_path, os.path.join(backup_path, filename))
            
            print(f"Current model backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            print(f"Error backing up model: {e}")
            return None
    
    def _train_new_model(self):
        """Train a new model using the main training script."""
        try:
            print("Training new model...")
            
            # Use the main training script
            predictor = CustomerChurnPredictor()
            results = predictor.run_complete_pipeline()
            
            if results and results.get('success', False):
                return {
                    'success': True,
                    'metrics': {
                        'f1_score': results.get('f1_score', 0.0),
                        'auc_score': results.get('auc_score', 0.0),
                        'precision': results.get('precision', 0.0),
                        'recall': results.get('recall', 0.0)
                    },
                    'model_path': results.get('model_path', ''),
                    'message': 'New model trained successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'Training failed - no results returned'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Training error: {str(e)}'
            }
    
    def _compare_models(self, current_metrics, new_metrics):
        """Compare current and new model performance."""
        try:
            if not current_metrics or not new_metrics:
                return False, "Missing metrics for comparison"
            
            # Check if new model is better
            improvements = []
            
            if new_metrics['f1_score'] > current_metrics['f1_score'] + self.config['min_improvement']:
                improvements.append(f"F1 improved: {current_metrics['f1_score']:.3f} -> {new_metrics['f1_score']:.3f}")
            
            if new_metrics['auc_score'] > current_metrics['auc_score'] + self.config['min_improvement']:
                improvements.append(f"AUC improved: {current_metrics['auc_score']:.3f} -> {new_metrics['auc_score']:.3f}")
            
            if new_metrics['precision'] > current_metrics['precision'] + self.config['min_improvement']:
                improvements.append(f"Precision improved: {current_metrics['precision']:.3f} -> {new_metrics['precision']:.3f}")
            
            if new_metrics['recall'] > current_metrics['recall'] + self.config['min_improvement']:
                improvements.append(f"Recall improved: {current_metrics['recall']:.3f} -> {new_metrics['recall']:.3f}")
            
            if improvements:
                return True, "; ".join(improvements)
            else:
                return False, "No significant improvement"
                
        except Exception as e:
            print(f"Error comparing models: {e}")
            return False, f"Comparison error: {str(e)}"
    
    def _log_retraining_event(self, event_type, details):
        """Log retraining events."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'event_type': event_type,
                'details': details
            }
            
            log_file = os.path.join(self.models_dir, 'retraining_log.json')
            
            # Load existing log or create new
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log = json.load(f)
            else:
                log = []
            
            log.append(log_entry)
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(log, f, indent=2)
            
            print(f"Logged: {event_type} - {details}")
            
        except Exception as e:
            print(f"Warning: Error logging retraining event: {e}")
    
    def retrain_model(self):
        """Main retraining method - checks if retraining is needed and performs it."""
        print("Starting model retraining process...")
        
        try:
            # Backup current model
            backup_path = self._backup_current_model()
            if not backup_path:
                print("Warning: Could not backup current model")
            
            # Get current model metrics
            current_metrics = self._get_current_model_metrics()
            
            # Train new model
            training_result = self._train_new_model()
            
            if not training_result['success']:
                print(f"Retraining failed: {training_result['error']}")
                self._log_retraining_event('retraining_failed', training_result['error'])
                return training_result
            
            # Compare models
            is_better, reason = self._compare_models(current_metrics, training_result['metrics'])
            
            if is_better:
                print(f"New model is better: {reason}")
                self._log_retraining_event('retraining_success', reason)
                return {
                    'success': True,
                    'message': f'Model retrained successfully: {reason}',
                    'metrics': training_result['metrics']
                }
            else:
                print(f"New model not better: {reason}")
                self._log_retraining_event('retraining_rejected', reason)
                return {
                    'success': False,
                    'message': f'Retraining rejected: {reason}',
                    'metrics': training_result['metrics']
                }
                
        except Exception as e:
            error_msg = f"Retraining error: {str(e)}"
            print(error_msg)
            self._log_retraining_event('retraining_error', error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def force_retrain(self):
        """Force retraining regardless of current performance."""
        print("Force retraining model...")
        
        try:
            # Train new model
            training_result = self._train_new_model()
            
            if training_result['success']:
                print("Force retraining completed successfully")
                self._log_retraining_event('force_retraining_success', 'Forced retraining completed')
                return training_result
            else:
                print(f"Force retraining failed: {training_result['error']}")
                self._log_retraining_event('force_retraining_failed', training_result['error'])
                return training_result
                
        except Exception as e:
            error_msg = f"Force retraining error: {str(e)}"
            print(error_msg)
            self._log_retraining_event('force_retraining_error', error_msg)
            return {
                'success': False,
                'error': error_msg
            }


def main():
    """Main function for testing."""
    retrainer = ModelRetrainer()
    
    # Test retraining
    result = retrainer.retrain_model()
    
    if result['success']:
        print("Model retraining completed successfully!")
    else:
        print("\nModel retraining failed!")


if __name__ == "__main__":
    main()
