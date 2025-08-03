# Monitoring and Drift Detection

Comprehensive monitoring system for Customer Churn Prediction. Detects data drift, concept drift, and tracks ongoing model performance.

## Files

### `drift_detection.py`
- **Purpose**: Detect data drift and concept drift
- **Features**:
  - Statistical drift detection for numerical features
  - Chi-square tests for categorical features
  - Performance-based concept drift detection
  - Configurable thresholds and significance levels

### `performance_monitor.py`
- **Purpose**: Track ongoing model performance
- **Features**:
  - Comprehensive performance metrics calculation
  - Performance history tracking
  - Trend analysis and degradation detection
  - Performance report generation

### `monitoring_orchestrator.py`
- **Purpose**: Coordinate all monitoring activities
- **Features**:
  - Unified monitoring interface
  - Comprehensive reporting
  - Actionable recommendations
  - Scheduled monitoring support

### `run_monitoring.py`
- **Purpose**: Run monitoring demonstrations and scheduled checks
- **Features**:
  - Demo mode with drift simulation
  - Scheduled monitoring mode
  - Command-line interface
  - Report generation

## Usage

### Run Monitoring Demo

```powershell
# Run monitoring demonstration
uv run python project_files/monitoring/run_monitoring.py --mode demo

# Run scheduled monitoring
uv run python project_files/monitoring/run_monitoring.py --mode scheduled
```

### Use Monitoring Components

```python
from project_files.monitoring import DriftDetector, PerformanceMonitor, MonitoringOrchestrator

# Initialize drift detector
detector = DriftDetector(reference_data, threshold=0.05)

# Initialize performance monitor
monitor = PerformanceMonitor()

# Initialize orchestrator
orchestrator = MonitoringOrchestrator(reference_data, reference_performance)
```

## Features

### Data Drift Detection
- ✅ **Statistical Tests**: Kolmogorov-Smirnov, Mann-Whitney U, Chi-square
- ✅ **Feature-wise Analysis**: Individual feature drift detection
- ✅ **Overall Drift Score**: Percentage of drifted features
- ✅ **Configurable Thresholds**: Adjustable significance levels

### Concept Drift Detection
- ✅ **Performance-based**: Uses F1 score degradation
- ✅ **Reference Comparison**: Compares against baseline performance
- ✅ **Degradation Thresholds**: Configurable performance drop limits
- ✅ **Actionable Alerts**: Clear recommendations for retraining

### Performance Monitoring
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC AUC
- ✅ **Historical Tracking**: Performance over time
- ✅ **Trend Analysis**: Improving, declining, or stable performance
- ✅ **Degradation Detection**: Automatic performance drop alerts

### Monitoring Orchestration
- ✅ **Unified Interface**: Single point for all monitoring activities
- ✅ **Comprehensive Reports**: Detailed monitoring summaries
- ✅ **Actionable Recommendations**: Clear next steps
- ✅ **Scheduled Monitoring**: Automated monitoring checks

## Available Metrics

### Drift Detection Metrics
- **Data Drift Score**: Percentage of features showing drift
- **Drifted Features**: List of features with detected drift
- **Statistical Significance**: P-values for drift tests
- **Drift Types**: Distribution shift, median shift, categorical drift

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Precision for each class
- **Recall**: Recall for each class
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve
- **Performance Trends**: Improving, declining, or stable

### Monitoring Reports
- **30-day Summary**: Long-term performance trends
- **7-day Summary**: Recent performance analysis
- **Drift Incidents**: Count of drift events
- **Average Performance**: Mean performance metrics

## 🔬 Drift Detection Methods

### Numerical Features
```python
# Kolmogorov-Smirnov test for distribution comparison
ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)

# Mann-Whitney U test for median comparison
mw_stat, mw_p_value = stats.mannwhitneyu(ref_data, curr_data)
```

### Categorical Features
```python
# Chi-square test for distribution comparison
chi2_stat, p_value = stats.chi2_contingency([ref_counts, curr_counts])
```

### Concept Drift
```python
# Performance degradation detection
degradation = reference_performance - current_performance
drift_detected = degradation > threshold
```

## 📈 Performance Tracking

### Historical Performance
```python
# Log performance metrics
performance_data = monitor.log_performance(y_true, y_pred, y_pred_proba)

# Get performance summary
summary = monitor.get_performance_summary(days=30)

# Analyze trends
trends = monitor.get_performance_trends(days=30)
```

### Performance Degradation
```python
# Detect performance degradation
degradation_results = monitor.detect_performance_degradation(current_f1)

# Set reference performance
monitor.set_reference_performance(reference_f1)
```

## 🚀 Quick Start

### 1. Run Monitoring Demo
```powershell
uv run python project_files/monitoring/run_monitoring.py --mode demo
```

### 2. Set Up Scheduled Monitoring
```powershell
# For Windows Task Scheduler or cron
uv run python project_files/monitoring/run_monitoring.py --mode scheduled
```

### 3. Use Monitoring in Code
```python
from project_files.monitoring import MonitoringOrchestrator

# Initialize orchestrator
orchestrator = MonitoringOrchestrator(
    reference_data=X_train,
    reference_performance=0.75
)

# Run monitoring check
results = orchestrator.run_monitoring_check(
    current_data=X_test,
    y_true=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba
)
```

## 📋 Monitoring Workflow

### 1. Initial Setup
```python
# Train reference model and get baseline performance
reference_model.fit(X_train, y_train)
reference_performance = f1_score(y_test, y_pred, average='weighted')

# Initialize monitoring system
orchestrator = MonitoringOrchestrator(X_train, reference_performance)
```

### 2. Regular Monitoring
```python
# Load current data
current_data = load_current_data()

# Get model predictions
predictions = model.predict(current_data)
probabilities = model.predict_proba(current_data)

# Run monitoring check
results = orchestrator.run_monitoring_check(
    current_data=current_data,
    y_true=true_labels,
    y_pred=predictions,
    y_pred_proba=probabilities
)
```

### 3. Review Results
```python
# Check for drift
if results['drift_detection']['overall_drift']['overall_drift_detected']:
    print("Drift detected! Consider retraining.")

# Check performance
if results['performance']['degradation_detected']:
    print("Performance degradation detected!")

# Review recommendations
for rec in results['recommendations']:
    print(f"Recommendation: {rec}")
```

## 🗂️ File Structure

```
project_files/monitoring/
├── __init__.py                    # Package initialization
├── drift_detection.py             # Data and concept drift detection
├── performance_monitor.py          # Performance tracking and analysis
├── monitoring_orchestrator.py     # Unified monitoring coordination
├── run_monitoring.py              # Monitoring runner and demo
├── README.md                      # This documentation
├── performance_metrics.json        # Performance history
├── monitoring_results.json         # Monitoring results
├── performance_report.json         # Performance analysis report
└── monitoring_report.json          # Comprehensive monitoring report
```

## 📊 Report Examples

### Performance Report
```json
{
  "generated_at": "2024-01-15T10:30:00",
  "summary_30_days": {
    "period_days": 30,
    "total_samples": 15000,
    "measurement_count": 30,
    "metrics_summary": {
      "f1_score": {
        "mean": 0.752,
        "std": 0.023,
        "trend": "stable"
      }
    }
  }
}
```

### Drift Detection Report
```json
{
  "drift_detection": {
    "data_drift": {
      "detected": true,
      "score": 0.25,
      "drifted_features": ["total_sessions", "total_events"]
    },
    "concept_drift": {
      "detected": false,
      "score": 0.0
    }
  },
  "recommendations": [
    "Data drift detected in 2 features. Monitor model performance closely."
  ]
}
```

## 🔄 Integration

### With Existing Pipeline
```python
# In customer_churn_prediction.py
from project_files.monitoring import MonitoringOrchestrator

# After model training
orchestrator = MonitoringOrchestrator(X_train, best_f1_score)

# During prediction
results = orchestrator.run_monitoring_check(
    current_data=new_data,
    y_true=ground_truth,
    y_pred=predictions
)
```

### With FastAPI
```python
# In fastapi_app.py
from project_files.monitoring import PerformanceMonitor

# Monitor API performance
monitor = PerformanceMonitor()
monitor.log_performance(y_true, y_pred, metadata={'endpoint': '/predict'})
```

### With MLflow
```python
# Log monitoring results to MLflow
import mlflow
from project_files.monitoring import MonitoringOrchestrator

with mlflow.start_run():
    results = orchestrator.run_monitoring_check(...)
    mlflow.log_metrics(results['performance']['current_metrics'])
    mlflow.log_params({'drift_detected': results['drift_detection']['overall_drift']['overall_drift_detected']})
```

## 📖 Best Practices

1. **Set Appropriate Thresholds**: Adjust drift detection sensitivity based on your domain
2. **Regular Monitoring**: Run monitoring checks at consistent intervals
3. **Reference Data**: Use representative training data as reference
4. **Performance Baselines**: Establish clear performance expectations
5. **Actionable Alerts**: Ensure recommendations are specific and actionable

## 🛠️ Customization

### Adding New Drift Detection Methods
```python
# In drift_detection.py
def _detect_custom_drift(self, column: str, current_data: pd.DataFrame) -> bool:
    # Implement custom drift detection logic
    # Return True if drift detected, False otherwise
    pass
```

### Custom Performance Metrics
```python
# In performance_monitor.py
def calculate_custom_metrics(self, y_true, y_pred, y_pred_proba):
    metrics = super().calculate_performance_metrics(y_true, y_pred, y_pred_proba)
    
    # Add custom metrics
    from sklearn.metrics import balanced_accuracy_score
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    return metrics
```

## 📖 Documentation

- **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
- **SciPy Statistical Tests**: https://docs.scipy.org/doc/scipy/reference/stats.html
- **Pandas Data Analysis**: https://pandas.pydata.org/docs/ 