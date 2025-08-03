"""
Monitoring Orchestrator for Customer Churn Prediction.
Coordinates drift detection and performance monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os

from .drift_detection import DriftDetector
from .performance_monitor import PerformanceMonitor


class MonitoringOrchestrator:
    """Orchestrates monitoring activities for the churn prediction model."""
    
    def __init__(self, 
                 reference_data: pd.DataFrame,
                 reference_performance: float,
                 monitoring_dir: str = "project_files/monitoring"):
        """
        Initialize monitoring orchestrator.
        
        Args:
            reference_data: Reference dataset for drift detection
            reference_performance: Reference performance metric (F1 score)
            monitoring_dir: Directory for monitoring files
        """
        self.monitoring_dir = monitoring_dir
        os.makedirs(monitoring_dir, exist_ok=True)
        
        # Initialize components
        self.drift_detector = DriftDetector(reference_data)
        self.performance_monitor = PerformanceMonitor(
            metrics_file=f"{monitoring_dir}/performance_metrics.json"
        )
        
        # Set reference performance
        self.performance_monitor.set_reference_performance(reference_performance)
        
        # Monitoring results
        self.monitoring_results = []
    
    def run_monitoring_check(self, 
                           current_data: pd.DataFrame,
                           y_true: List,
                           y_pred: List,
                           y_pred_proba: Optional[List] = None,
                           metadata: Optional[Dict] = None) -> Dict:
        """
        Run comprehensive monitoring check.
        
        Args:
            current_data: Current feature data
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Dictionary with comprehensive monitoring results
        """
        print("🔍 Running Monitoring Check...")
        print("=" * 40)
        
        # 1. Performance monitoring
        print("📊 Checking performance metrics...")
        performance_data = self.performance_monitor.log_performance(
            y_true, y_pred, y_pred_proba, metadata
        )
        
        # 2. Data drift detection
        print("📈 Checking for data drift...")
        data_drift_results = self.drift_detector.detect_data_drift(current_data)
        
        # 3. Concept drift detection
        print("🧠 Checking for concept drift...")
        reference_performance = self.performance_monitor.get_reference_performance()
        concept_drift_results = self.drift_detector.detect_concept_drift(
            y_true, y_pred, reference_performance
        )
        
        # 4. Performance degradation detection
        print("📉 Checking for performance degradation...")
        current_f1 = performance_data['metrics']['f1_score']
        degradation_results = self.performance_monitor.detect_performance_degradation(current_f1)
        
        # 5. Generate comprehensive report
        monitoring_report = self._generate_monitoring_report(
            performance_data,
            data_drift_results,
            concept_drift_results,
            degradation_results,
            metadata
        )
        
        # 6. Store results
        self.monitoring_results.append(monitoring_report)
        self._save_monitoring_results()
        
        # 7. Print summary
        self._print_monitoring_summary(monitoring_report)
        
        return monitoring_report
    
    def _generate_monitoring_report(self,
                                  performance_data: Dict,
                                  data_drift_results: Dict,
                                  concept_drift_results: Dict,
                                  degradation_results: Dict,
                                  metadata: Optional[Dict]) -> Dict:
        """Generate comprehensive monitoring report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'current_metrics': performance_data['metrics'],
                'sample_count': performance_data['sample_count'],
                'degradation_detected': degradation_results['degradation_detected'],
                'degradation_amount': degradation_results.get('degradation_amount', 0)
            },
            'drift_detection': {
                'data_drift': data_drift_results,
                'concept_drift': concept_drift_results,
                'overall_drift': self.drift_detector.get_drift_summary(
                    data_drift_results, concept_drift_results
                )
            },
            'recommendations': self._generate_recommendations(
                data_drift_results, concept_drift_results, degradation_results
            ),
            'metadata': metadata or {}
        }
    
    def _generate_recommendations(self,
                                 data_drift_results: Dict,
                                 concept_drift_results: Dict,
                                 degradation_results: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data drift recommendations
        if data_drift_results['drift_detected']:
            recommendations.append(
                f"Data drift detected in {len(data_drift_results['drifted_features'])} features. "
                "Monitor model performance closely."
            )
        
        # Concept drift recommendations
        if concept_drift_results['drift_detected']:
            recommendations.append(
                f"Concept drift detected. Performance degradation: "
                f"{concept_drift_results['performance_degradation']:.4f}. "
                "Consider model retraining."
            )
        
        # Performance degradation recommendations
        if degradation_results['degradation_detected']:
            recommendations.append(
                f"Performance degradation detected: {degradation_results['degradation_amount']:.4f}. "
                f"Recommendation: {degradation_results['recommendation']}"
            )
        
        # No issues detected
        if not any([
            data_drift_results['drift_detected'],
            concept_drift_results['drift_detected'],
            degradation_results['degradation_detected']
        ]):
            recommendations.append("No significant issues detected. Continue monitoring.")
        
        return recommendations
    
    def _print_monitoring_summary(self, report: Dict):
        """Print monitoring summary to console."""
        print("\n📋 Monitoring Summary")
        print("-" * 40)
        
        # Performance summary
        metrics = report['performance']['current_metrics']
        print(f"📊 Current Performance:")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        
        # Drift summary
        drift_summary = report['drift_detection']['overall_drift']
        print(f"\n🔍 Drift Detection:")
        print(f"   Data Drift: {'✅ Detected' if drift_summary['data_drift']['detected'] else '❌ None'}")
        print(f"   Concept Drift: {'✅ Detected' if drift_summary['concept_drift']['detected'] else '❌ None'}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 40)
    
    def _save_monitoring_results(self):
        """Save monitoring results to file."""
        results_file = f"{self.monitoring_dir}/monitoring_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(self.monitoring_results, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save monitoring results: {e}")
    
    def get_monitoring_summary(self, days: int = 30) -> Dict:
        """Get monitoring summary for the last N days."""
        if not self.monitoring_results:
            return {'error': 'No monitoring data available'}
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [
            r for r in self.monitoring_results 
            if datetime.fromisoformat(r['timestamp']) > cutoff_date
        ]
        
        if not recent_results:
            return {'error': f'No monitoring data in last {days} days'}
        
        # Calculate summary
        summary = {
            'period_days': days,
            'total_checks': len(recent_results),
            'drift_incidents': {
                'data_drift': sum(1 for r in recent_results 
                                if r['drift_detection']['data_drift']['drift_detected']),
                'concept_drift': sum(1 for r in recent_results 
                                   if r['drift_detection']['concept_drift']['drift_detected']),
                'performance_degradation': sum(1 for r in recent_results 
                                             if r['performance']['degradation_detected'])
            },
            'average_performance': {
                'f1_score': np.mean([r['performance']['current_metrics']['f1_score'] 
                                   for r in recent_results]),
                'accuracy': np.mean([r['performance']['current_metrics']['accuracy'] 
                                   for r in recent_results])
            }
        }
        
        return summary
    
    def export_monitoring_report(self, output_file: str = "project_files/monitoring/monitoring_report.json"):
        """Export comprehensive monitoring report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary_30_days': self.get_monitoring_summary(30),
            'summary_7_days': self.get_monitoring_summary(7),
            'performance_trends': self.performance_monitor.get_performance_trends(30),
            'total_monitoring_checks': len(self.monitoring_results),
            'reference_performance': self.performance_monitor.get_reference_performance()
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✅ Monitoring report exported to: {output_file}")
        except Exception as e:
            print(f"❌ Error exporting monitoring report: {e}")
        
        return report
    
    def run_scheduled_monitoring(self, 
                               data_loader_func,
                               model_predict_func,
                               schedule_interval: str = "daily") -> Dict:
        """
        Run scheduled monitoring check.
        
        Args:
            data_loader_func: Function to load current data
            model_predict_func: Function to get model predictions
            schedule_interval: Monitoring schedule (daily, weekly, monthly)
            
        Returns:
            Dictionary with monitoring results
        """
        print(f"🕐 Running {schedule_interval} monitoring check...")
        
        try:
            # Load current data
            current_data = data_loader_func()
            
            # Get predictions (assuming data includes labels for monitoring)
            predictions = model_predict_func(current_data)
            
            # Extract true labels and predictions
            y_true = current_data['is_churned'].tolist()  # Assuming this column exists
            y_pred = predictions['predictions']
            y_pred_proba = predictions.get('probabilities', None)
            
            # Run monitoring check
            monitoring_results = self.run_monitoring_check(
                current_data.drop(columns=['is_churned']),  # Remove target for drift detection
                y_true, y_pred, y_pred_proba,
                metadata={'schedule_interval': schedule_interval}
            )
            
            return monitoring_results
            
        except Exception as e:
            print(f"❌ Error in scheduled monitoring: {e}")
            return {'error': str(e)} 