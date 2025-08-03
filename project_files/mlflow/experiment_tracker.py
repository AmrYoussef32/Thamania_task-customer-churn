"""
MLflow experiment tracker for Customer Churn Prediction.
Simple experiment tracking and model comparison.
"""

import mlflow
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .mlflow_config import get_mlflow_config, log_experiment_results


class ExperimentTracker:
    """Simple experiment tracker for MLflow."""
    
    def __init__(self):
        self.config = get_mlflow_config()
        self.current_run = None
    
    def start_experiment(self, experiment_name: str = None):
        """Start a new experiment."""
        if experiment_name:
            self.config.experiment_name = experiment_name
            self.config._setup_mlflow()
        
        self.current_run = self.config.start_run()
        return self.current_run
    
    def log_model_experiment(
        self,
        model,
        model_name: str,
        X_train,
        X_test,
        y_train,
        y_test,
        params: Dict[str, Any],
        additional_metrics: Dict[str, float] = None
    ):
        """Log a complete model experiment."""
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log to MLflow
        run_id = log_experiment_results(
            params=params,
            metrics=metrics,
            model=model,
            model_name=model_name,
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Create and log confusion matrix
        self._log_confusion_matrix(y_test, y_pred, run_id)
        
        return run_id, metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba)
        }
        
        # Add per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):
                for metric_name, value in class_metrics.items():
                    if metric_name != 'support':
                        metrics[f"{class_name}_{metric_name}"] = value
        
        return metrics
    
    def _log_confusion_matrix(self, y_true, y_pred, run_id):
        """Create and log confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save and log
        cm_path = f"project_files/mlflow/confusion_matrix_{run_id}.png"
        plt.savefig(cm_path)
        plt.close()
        
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(cm_path, "confusion_matrix")
    
    def compare_models(self, model_results: List[Dict]) -> pd.DataFrame:
        """Compare multiple models."""
        comparison_data = []
        
        for result in model_results:
            comparison_data.append({
                'model_name': result['model_name'],
                'run_id': result['run_id'],
                **result['metrics']
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model_info(self, metric_name: str = "f1_score") -> Optional[Dict]:
        """Get information about the best model."""
        best_run_id = self.config.get_best_model(metric_name)
        
        if not best_run_id:
            return None
        
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(best_run_id)
        
        return {
            'run_id': best_run_id,
            'model_name': run.data.params.get('model_name', 'unknown'),
            'metrics': run.data.metrics,
            'parameters': run.data.params,
            'artifact_uri': run.info.artifact_uri
        }
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        client = mlflow.tracking.MlflowClient()
        experiments = client.list_experiments()
        
        experiment_list = []
        for exp in experiments:
            runs = client.search_runs([exp.experiment_id])
            experiment_list.append({
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'runs_count': len(runs),
                'artifact_location': exp.artifact_location
            })
        
        return experiment_list
    
    def export_experiment_results(self, output_path: str = "project_files/mlflow/experiment_results.csv"):
        """Export all experiment results to CSV."""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.config.experiment_name)
        
        if not experiment:
            print("No experiments found.")
            return
        
        runs = client.search_runs([experiment.experiment_id])
        
        results = []
        for run in runs:
            result = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', ''),
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'status': run.info.status
            }
            
            # Add parameters
            for key, value in run.data.params.items():
                result[f"param_{key}"] = value
            
            # Add metrics
            for key, value in run.data.metrics.items():
                result[f"metric_{key}"] = value
            
            results.append(result)
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"✅ Experiment results exported to: {output_path}")
        
        return df


# Global experiment tracker
experiment_tracker = ExperimentTracker()


def get_experiment_tracker() -> ExperimentTracker:
    """Get the global experiment tracker."""
    return experiment_tracker 