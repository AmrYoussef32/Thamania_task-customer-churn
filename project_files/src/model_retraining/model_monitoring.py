#!/usr/bin/env python3
"""
Model Monitoring System

Monitors model performance and detects when retraining is needed.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Import the main training script
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from customer_churn_prediction import CustomerChurnPredictor


class ModelMonitor:
    """Monitors model performance and determines when retraining is needed."""
    
    def __init__(self, config_path="../../config/retraining_config.json"):
        """Initialize the model monitor with configuration."""
        self.config = self._load_config(config_path)
        self.models_dir = self.config.get('models_dir', '../../models')
        self.data_path = self.config.get('data_path', '../../data/customer_churn_mini.json')
        
        # Performance thresholds
        self.f1_threshold = self.config.get('f1_threshold', 0.7)
        self.auc_threshold = self.config.get('auc_threshold', 0.6)
        self.precision_threshold = self.config.get('precision_threshold', 0.5)
        self.recall_threshold = self.config.get('recall_threshold', 0.6)
        self.drift_threshold = self.config.get('drift_threshold', 0.3)
    
    def _load_config(self, config_path):
        """Load monitoring configuration."""
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
            'models_dir': '../../models'
        }
    
    def _load_current_model(self):
        """Load the current trained model."""
        try:
            # Find the latest model file
            model_files = [f for f in os.listdir(self.models_dir) 
                         if f.startswith('churn_model_') and f.endswith('.pkl')]
            
            if not model_files:
                print("No trained model found")
                return None, None, None
            
            # Get the latest model file
            latest_model_file = sorted(model_files)[-1]
            model_path = os.path.join(self.models_dir, latest_model_file)
            
            # Load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load preprocessing
            preprocessing_files = [f for f in os.listdir(self.models_dir) 
                                if f.startswith('preprocessing_') and f.endswith('.pkl')]
            preprocessing = None
            if preprocessing_files:
                latest_preprocessing_file = sorted(preprocessing_files)[-1]
                preprocessing_path = os.path.join(self.models_dir, latest_preprocessing_file)
                with open(preprocessing_path, 'rb') as f:
                    preprocessing = pickle.load(f)
            
            # Load feature columns
            feature_files = [f for f in os.listdir(self.models_dir) 
                           if f.startswith('feature_columns_') and f.endswith('.json')]
            feature_columns = None
            if feature_files:
                latest_feature_file = sorted(feature_files)[-1]
                feature_path = os.path.join(self.models_dir, latest_feature_file)
                with open(feature_path, 'r') as f:
                    feature_columns = json.load(f)
            
            print(f"Loaded current model: {model_path}")
            return model, preprocessing, feature_columns
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None
    
    def _prepare_test_data(self):
        """Prepare test data for model evaluation."""
        try:
            # Load the dataset
            df = pd.read_json(self.data_path)
            
            # Use the main training script to prepare data
            predictor = CustomerChurnPredictor()
            
            # Get user features and targets
            user_features, targets = predictor.prepare_modeling_data(df)
            
            if user_features is None or targets is None:
                print("Error: Could not prepare test data")
                return None, None
            
            # Split into train/test for evaluation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                user_features, targets, test_size=0.2, random_state=42, stratify=targets
            )
            
            return X_test, y_test
            
        except Exception as e:
            print(f"Error preparing test data: {e}")
            return None, None
    
    def _create_features_from_data(self, df):
        """Create features from raw data."""
        try:
            # Use the main training script to create features
            predictor = CustomerChurnPredictor()
            
            # Create user features
            user_features = predictor.create_user_features(df)
            
            if user_features is None or user_features.empty:
                print("Error: Could not create features from data")
                return None
            
            return user_features
            
        except Exception as e:
            print(f"Error creating features: {e}")
            return None
    
    def _evaluate_model_performance(self, model, X_test, y_test):
        """Evaluate model performance on test data."""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            return {
                'f1_score': f1,
                'auc_score': auc,
                'precision': precision,
                'recall': recall
            }
            
        except Exception as e:
            print(f"Error checking performance: {e}")
            return None
    
    def _detect_data_drift(self, reference_data, current_data):
        """Detect data drift between reference and current data."""
        try:
            # Simple drift detection - compare basic statistics
            drift_score = 0.0
            drifted_features = []
            
            for column in reference_data.columns:
                if column in current_data.columns:
                    # Compare means for numerical features
                    if reference_data[column].dtype in ['int64', 'float64']:
                        ref_mean = reference_data[column].mean()
                        curr_mean = current_data[column].mean()
                        
                        # Calculate relative difference
                        if ref_mean != 0:
                            relative_diff = abs(curr_mean - ref_mean) / abs(ref_mean)
                            if relative_diff > self.drift_threshold:
                                drifted_features.append(column)
                                drift_score += 1
            
            # Normalize drift score
            if len(reference_data.columns) > 0:
                drift_score = drift_score / len(reference_data.columns)
            
            return {
                'drift_detected': drift_score > self.drift_threshold,
                'drift_score': drift_score,
                'drifted_features': drifted_features
            }
            
        except Exception as e:
            print(f"Error detecting data drift: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drifted_features': []
            }
    
    def check_if_retraining_needed(self):
        """Check if model retraining is needed based on performance and drift."""
        try:
            # Load current model
            model, preprocessing, feature_columns = self._load_current_model()
            
            if model is None:
                print("Retraining needed: No current model found")
                return True, ["No trained model available"]
            
            # Prepare test data
            X_test, y_test = self._prepare_test_data()
            
            if X_test is None:
                print("Error: Could not load test data")
                return True, ["Could not prepare test data"]
            
            # Check model performance
            performance = self._evaluate_model_performance(model, X_test, y_test)
            
            if performance is None:
                print("Error: Could not create features")
                return True, ["Could not evaluate model performance"]
            
            # Check for performance degradation
            reasons = []
            
            if performance['f1_score'] < self.f1_threshold:
                reasons.append(f"F1 score below threshold: {performance['f1_score']:.3f} < {self.f1_threshold}")
            
            if performance['auc_score'] < self.auc_threshold:
                reasons.append(f"AUC score below threshold: {performance['auc_score']:.3f} < {self.auc_threshold}")
            
            if performance['precision'] < self.precision_threshold:
                reasons.append(f"Precision below threshold: {performance['precision']:.3f} < {self.precision_threshold}")
            
            if performance['recall'] < self.recall_threshold:
                reasons.append(f"Recall below threshold: {performance['recall']:.3f} < {self.recall_threshold}")
            
            # Check for data drift (simplified)
            # In a real implementation, you would compare current data with training data
            # For now, we'll skip drift detection for simplicity
            
            if reasons:
                print(f"Retraining needed: {', '.join(reasons)}")
                return True, reasons
            else:
                print("No retraining needed: Model performing well")
                return False, []
                
        except Exception as e:
            print(f"Error in retraining check: {e}")
            return True, [f"Error during monitoring: {str(e)}"]
    
    def get_model_performance(self):
        """Get current model performance metrics."""
        try:
            # Load current model
            model, preprocessing, feature_columns = self._load_current_model()
            
            if model is None:
                return None
            
            # Prepare test data
            X_test, y_test = self._prepare_test_data()
            
            if X_test is None:
                return None
            
            # Evaluate performance
            performance = self._evaluate_model_performance(model, X_test, y_test)
            
            return performance
            
        except Exception as e:
            print(f"Error getting model performance: {e}")
            return None


def main():
    """Main function for testing."""
    monitor = ModelMonitor()
    
    # Check if retraining is needed
    retraining_needed, reasons = monitor.check_if_retraining_needed()
    
    if retraining_needed:
        print(f"\nRetraining triggered: {reasons}")
    else:
        print("\nModel is performing well, no retraining needed")


if __name__ == "__main__":
    main()
