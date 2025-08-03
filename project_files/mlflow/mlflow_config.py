"""
MLflow configuration for Customer Churn Prediction project.
Simple and organized MLflow setup for experiment tracking.
"""

import os
import mlflow
from datetime import datetime
from typing import Dict, Any, Optional


class MLflowConfig:
    """Simple MLflow configuration for the project."""
    
    def __init__(self, experiment_name: str = "customer_churn_prediction"):
        self.experiment_name = experiment_name
        self.tracking_uri = "sqlite:///project_files/mlflow/mlflow.db"
        self.artifact_location = "./project_files/mlflow/artifacts"
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking and experiment."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_location
            )
        
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name)
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, model_name: str = "churn_model"):
        """Log model to MLflow."""
        mlflow.sklearn.log_model(model, model_name)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact to MLflow."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def get_best_model(self, metric_name: str = "f1_score", greater_is_better: bool = True):
        """Get the best model based on a metric."""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            return None
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'DESC' if greater_is_better else 'ASC'}"]
        )
        
        if not runs:
            return None
        
        best_run = runs[0]
        return best_run.info.run_id
    
    def load_model(self, run_id: str, model_name: str = "churn_model"):
        """Load a model from MLflow."""
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.sklearn.load_model(model_uri)


# Global MLflow instance
mlflow_config = MLflowConfig()


def get_mlflow_config() -> MLflowConfig:
    """Get the global MLflow configuration."""
    return mlflow_config


def log_experiment_results(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model,
    model_name: str = "churn_model",
    run_name: Optional[str] = None
):
    """Log complete experiment results to MLflow."""
    config = get_mlflow_config()
    
    with config.start_run(run_name) as run:
        # Log parameters
        config.log_parameters(params)
        
        # Log metrics
        config.log_metrics(metrics)
        
        # Log model
        config.log_model(model, model_name)
        
        # Log additional artifacts
        config.log_artifact("project_files/models/", "models")
        config.log_artifact("project_files/data/", "data")
        
        print(f"✅ Experiment logged to MLflow: {run.info.run_id}")
        return run.info.run_id


def get_best_model_uri(metric_name: str = "f1_score") -> Optional[str]:
    """Get the URI of the best model."""
    config = get_mlflow_config()
    best_run_id = config.get_best_model(metric_name)
    
    if best_run_id:
        return f"runs:/{best_run_id}/churn_model"
    return None


def load_best_model(metric_name: str = "f1_score"):
    """Load the best model from MLflow."""
    config = get_mlflow_config()
    best_run_id = config.get_best_model(metric_name)
    
    if best_run_id:
        return config.load_model(best_run_id)
    return None 