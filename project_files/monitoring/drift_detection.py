"""
Drift Detection for Customer Churn Prediction.
Simple monitoring system for data drift and concept drift detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DriftDetector:
    """Simple drift detection for monitoring model performance."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset for comparison
            threshold: Significance threshold for drift detection
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._calculate_reference_stats()
    
    def _calculate_reference_stats(self) -> Dict:
        """Calculate reference statistics for drift detection."""
        stats_dict = {}
        
        for column in self.reference_data.select_dtypes(include=[np.number]).columns:
            stats_dict[column] = {
                'mean': self.reference_data[column].mean(),
                'std': self.reference_data[column].std(),
                'median': self.reference_data[column].median(),
                'q25': self.reference_data[column].quantile(0.25),
                'q75': self.reference_data[column].quantile(0.75)
            }
        
        # For categorical columns
        for column in self.reference_data.select_dtypes(include=['object', 'category']).columns:
            value_counts = self.reference_data[column].value_counts(normalize=True)
            stats_dict[column] = {
                'distribution': value_counts.to_dict(),
                'unique_count': self.reference_data[column].nunique()
            }
        
        return stats_dict
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare
            
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'drifted_features': [],
            'details': {}
        }
        
        total_features = 0
        drifted_features = 0
        
        for column in current_data.columns:
            if column not in self.reference_stats:
                continue
            
            total_features += 1
            drift_detected = False
            
            if column in self.reference_data.select_dtypes(include=[np.number]).columns:
                # Numerical feature drift detection
                drift_detected = self._detect_numerical_drift(column, current_data)
            else:
                # Categorical feature drift detection
                drift_detected = self._detect_categorical_drift(column, current_data)
            
            if drift_detected:
                drifted_features += 1
                drift_results['drifted_features'].append(column)
                drift_results['details'][column] = {
                    'drift_detected': True,
                    'drift_type': 'distribution_shift'
                }
            else:
                drift_results['details'][column] = {
                    'drift_detected': False,
                    'drift_type': 'no_drift'
                }
        
        # Calculate overall drift score
        if total_features > 0:
            drift_score = drifted_features / total_features
            drift_results['drift_score'] = drift_score
            drift_results['drift_detected'] = drift_score > self.threshold
        
        return drift_results
    
    def _detect_numerical_drift(self, column: str, current_data: pd.DataFrame) -> bool:
        """Detect drift in numerical features using statistical tests."""
        try:
            ref_data = self.reference_data[column].dropna()
            curr_data = current_data[column].dropna()
            
            if len(ref_data) < 10 or len(curr_data) < 10:
                return False
            
            # Kolmogorov-Smirnov test for distribution comparison
            ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
            
            # Mann-Whitney U test for median comparison
            mw_stat, mw_p_value = stats.mannwhitneyu(ref_data, curr_data, alternative='two-sided')
            
            # Check if either test indicates significant drift
            return p_value < self.threshold or mw_p_value < self.threshold
            
        except Exception:
            return False
    
    def _detect_categorical_drift(self, column: str, current_data: pd.DataFrame) -> bool:
        """Detect drift in categorical features using chi-square test."""
        try:
            ref_dist = self.reference_stats[column]['distribution']
            curr_dist = current_data[column].value_counts(normalize=True).to_dict()
            
            # Create contingency table for chi-square test
            all_categories = set(ref_dist.keys()) | set(curr_dist.keys())
            
            if len(all_categories) < 2:
                return False
            
            # Prepare data for chi-square test
            ref_counts = []
            curr_counts = []
            
            for category in all_categories:
                ref_counts.append(ref_dist.get(category, 0) * len(self.reference_data))
                curr_counts.append(curr_dist.get(category, 0) * len(current_data))
            
            # Chi-square test
            chi2_stat, p_value = stats.chi2_contingency([ref_counts, curr_counts])[:2]
            
            return p_value < self.threshold
            
        except Exception:
            return False
    
    def detect_concept_drift(self, 
                           y_true: List, 
                           y_pred: List, 
                           reference_performance: float) -> Dict:
        """
        Detect concept drift based on performance degradation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            reference_performance: Reference performance metric
            
        Returns:
            Dictionary with concept drift detection results
        """
        if len(y_true) < 10:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'current_performance': 0.0,
                'performance_degradation': 0.0,
                'reason': 'insufficient_data'
            }
        
        # Calculate current performance
        current_accuracy = accuracy_score(y_true, y_pred)
        current_f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Use F1 score as primary metric
        current_performance = current_f1
        performance_degradation = reference_performance - current_performance
        
        # Detect concept drift based on performance degradation
        drift_detected = performance_degradation > self.threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_score': max(0, performance_degradation),
            'current_performance': current_performance,
            'performance_degradation': performance_degradation,
            'current_accuracy': current_accuracy,
            'current_f1': current_f1,
            'reason': 'performance_degradation' if drift_detected else 'no_drift'
        }
    
    def get_drift_summary(self, data_drift_results: Dict, concept_drift_results: Dict) -> Dict:
        """Generate a summary of drift detection results."""
        return {
            'overall_drift_detected': data_drift_results['drift_detected'] or concept_drift_results['drift_detected'],
            'data_drift': {
                'detected': data_drift_results['drift_detected'],
                'score': data_drift_results['drift_score'],
                'drifted_features': data_drift_results['drifted_features']
            },
            'concept_drift': {
                'detected': concept_drift_results['drift_detected'],
                'score': concept_drift_results['drift_score'],
                'performance_degradation': concept_drift_results['performance_degradation']
            },
            'recommendation': self._get_recommendation(data_drift_results, concept_drift_results)
        }
    
    def _get_recommendation(self, data_drift_results: Dict, concept_drift_results: Dict) -> str:
        """Generate recommendation based on drift detection results."""
        if data_drift_results['drift_detected'] and concept_drift_results['drift_detected']:
            return "High drift detected. Consider retraining model with new data."
        elif data_drift_results['drift_detected']:
            return "Data drift detected. Monitor performance closely."
        elif concept_drift_results['drift_detected']:
            return "Concept drift detected. Consider retraining model."
        else:
            return "No significant drift detected. Continue monitoring." 