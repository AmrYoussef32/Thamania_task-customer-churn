"""
Performance Monitoring for Customer Churn Prediction.
Simple monitoring system for tracking ongoing model performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class PerformanceMonitor:
    """Simple performance monitoring for model tracking."""
    
    def __init__(self, metrics_file: str = "project_files/monitoring/performance_metrics.json"):
        """
        Initialize performance monitor.
        
        Args:
            metrics_file: File to store performance metrics
        """
        self.metrics_file = metrics_file
        self.metrics_history = self._load_metrics_history()
        self.reference_performance = None
        
        # Create monitoring directory if it doesn't exist
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    def _load_metrics_history(self) -> List[Dict]:
        """Load existing metrics history from file."""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def _save_metrics_history(self):
        """Save metrics history to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save metrics history: {e}")
    
    def calculate_performance_metrics(self, y_true: List, y_pred: List, y_pred_proba: Optional[List] = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {}
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC if probabilities are available
        if y_pred_proba is not None and len(y_pred_proba) > 0:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except Exception:
                metrics['roc_auc'] = 0.0
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Add per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):
                for metric_name, value in class_metrics.items():
                    if metric_name != 'support':
                        metrics[f"{class_name}_{metric_name}"] = value
        
        return metrics
    
    def log_performance(self, 
                       y_true: List, 
                       y_pred: List, 
                       y_pred_proba: Optional[List] = None,
                       metadata: Optional[Dict] = None) -> Dict:
        """
        Log performance metrics with timestamp.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Dictionary with logged performance data
        """
        metrics = self.calculate_performance_metrics(y_true, y_pred, y_pred_proba)
        
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'sample_count': len(y_true),
            'metadata': metadata or {}
        }
        
        # Add to history
        self.metrics_history.append(performance_data)
        
        # Save to file
        self._save_metrics_history()
        
        return performance_data
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """
        Get performance summary for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with performance summary
        """
        if not self.metrics_history:
            return {'error': 'No performance data available'}
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]
        
        if not recent_metrics:
            return {'error': f'No performance data in last {days} days'}
        
        # Calculate summary statistics
        summary = {
            'period_days': days,
            'total_samples': sum(m['sample_count'] for m in recent_metrics),
            'measurement_count': len(recent_metrics),
            'metrics_summary': {}
        }
        
        # Calculate average metrics
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metric_names:
            values = [m['metrics'].get(metric, 0) for m in recent_metrics if m['metrics'].get(metric) is not None]
            if values:
                summary['metrics_summary'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': self._calculate_trend(values)
                }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def set_reference_performance(self, f1_score: float):
        """Set reference performance for drift detection."""
        self.reference_performance = f1_score
    
    def get_reference_performance(self) -> Optional[float]:
        """Get reference performance."""
        return self.reference_performance
    
    def detect_performance_degradation(self, current_f1: float, threshold: float = 0.05) -> Dict:
        """
        Detect performance degradation compared to reference.
        
        Args:
            current_f1: Current F1 score
            threshold: Degradation threshold
            
        Returns:
            Dictionary with degradation detection results
        """
        if self.reference_performance is None:
            return {
                'degradation_detected': False,
                'reason': 'no_reference_performance'
            }
        
        degradation = self.reference_performance - current_f1
        degradation_detected = degradation > threshold
        
        return {
            'degradation_detected': degradation_detected,
            'degradation_amount': degradation,
            'reference_performance': self.reference_performance,
            'current_performance': current_f1,
            'threshold': threshold,
            'recommendation': self._get_degradation_recommendation(degradation, threshold)
        }
    
    def _get_degradation_recommendation(self, degradation: float, threshold: float) -> str:
        """Generate recommendation based on performance degradation."""
        if degradation > threshold * 2:
            return "Severe performance degradation. Immediate retraining recommended."
        elif degradation > threshold:
            return "Performance degradation detected. Consider retraining soon."
        else:
            return "Performance is stable. Continue monitoring."
    
    def get_performance_trends(self, days: int = 30) -> Dict:
        """
        Analyze performance trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        summary = self.get_performance_summary(days)
        
        if 'error' in summary:
            return summary
        
        trends = {
            'overall_trend': 'stable',
            'metric_trends': {},
            'recommendations': []
        }
        
        # Analyze trends for each metric
        for metric, stats in summary['metrics_summary'].items():
            trends['metric_trends'][metric] = stats['trend']
        
        # Determine overall trend
        improving_metrics = sum(1 for trend in trends['metric_trends'].values() if trend == 'improving')
        declining_metrics = sum(1 for trend in trends['metric_trends'].values() if trend == 'declining')
        
        if declining_metrics > improving_metrics:
            trends['overall_trend'] = 'declining'
            trends['recommendations'].append("Overall performance is declining. Consider model retraining.")
        elif improving_metrics > declining_metrics:
            trends['overall_trend'] = 'improving'
            trends['recommendations'].append("Overall performance is improving. Continue current approach.")
        else:
            trends['overall_trend'] = 'stable'
            trends['recommendations'].append("Performance is stable. Continue monitoring.")
        
        return trends
    
    def export_performance_report(self, output_file: str = "project_files/monitoring/performance_report.json"):
        """Export comprehensive performance report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary_30_days': self.get_performance_summary(30),
            'summary_7_days': self.get_performance_summary(7),
            'trends': self.get_performance_trends(30),
            'reference_performance': self.reference_performance,
            'total_measurements': len(self.metrics_history)
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✅ Performance report exported to: {output_file}")
        except Exception as e:
            print(f"❌ Error exporting performance report: {e}")
        
        return report 