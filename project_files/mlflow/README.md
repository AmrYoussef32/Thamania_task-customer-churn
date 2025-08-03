# MLflow Integration

MLflow integration for Customer Churn Prediction project. Simple experiment tracking and model management.

## 📁 Files

### `mlflow_config.py`
- **Purpose**: Core MLflow configuration and setup
- **Features**:
  - SQLite backend for tracking
  - Local artifact storage
  - Simple experiment management
  - Model logging and loading
  - Best model retrieval

### `experiment_tracker.py`
- **Purpose**: Experiment tracking and model comparison
- **Features**:
  - Comprehensive metrics calculation
  - Confusion matrix generation
  - Model comparison utilities
  - Experiment results export
  - Best model identification

### `run_experiments.py`
- **Purpose**: Run MLflow experiments with different models
- **Features**:
  - Multiple model configurations
  - Automated experiment execution
  - Results comparison
  - Best model selection

### `start_mlflow_ui.ps1`
- **Purpose**: PowerShell script to start MLflow UI
- **Features**:
  - Easy UI startup
  - Port configuration
  - Error handling

## 🔧 Usage

### Setup MLflow

```powershell
# Install MLflow
uv add mlflow

# Start MLflow UI
.\project_files\mlflow\start_mlflow_ui.ps1
```

### Run Experiments

```powershell
# Run all experiments
uv run python project_files/mlflow/run_experiments.py

# Or from Python
python -c "
from project_files.mlflow import get_experiment_tracker
tracker = get_experiment_tracker()
# Use tracker methods
"
```

### Access MLflow UI

1. Start the UI: `.\project_files\mlflow\start_mlflow_ui.ps1`
2. Open browser: http://localhost:5000
3. View experiments, runs, and models

## 🎯 Features

### Experiment Tracking
- ✅ **Automatic run creation** with timestamps
- ✅ **Parameter logging** for all model configurations
- ✅ **Metric tracking** (accuracy, precision, recall, F1, ROC AUC)
- ✅ **Model artifacts** storage and versioning
- ✅ **Confusion matrix** visualization

### Model Management
- ✅ **Model logging** with MLflow format
- ✅ **Best model retrieval** based on metrics
- ✅ **Model loading** from MLflow runs
- ✅ **Model comparison** across experiments

### Visualization
- ✅ **MLflow UI** for experiment browsing
- ✅ **Confusion matrices** for each model
- ✅ **Metric comparison** tables
- ✅ **Experiment results** export to CSV

## 📊 Available Metrics

### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Precision for each class
- **Recall**: Recall for each class
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve
- **Log Loss**: Logarithmic loss

### Per-Class Metrics
- **Class 0 (Not Churned)**: Precision, recall, F1
- **Class 1 (Churned)**: Precision, recall, F1

## 🔬 Experiment Types

### Default Experiments
1. **LogisticRegression**: Linear model with regularization
2. **RandomForest**: Ensemble of decision trees
3. **GradientBoosting**: Sequential boosting model

### Custom Experiments
```python
from project_files.mlflow import get_experiment_tracker
from sklearn.svm import SVC

tracker = get_experiment_tracker()

# Custom experiment
svm_model = SVC(probability=True, random_state=42)
run_id, metrics = tracker.log_model_experiment(
    model=svm_model,
    model_name="SVM",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    params={'C': 1.0, 'kernel': 'rbf'}
)
```

## 📈 Model Comparison

### Automatic Comparison
```python
# Compare all models
comparison_df = tracker.compare_models(results)
print(comparison_df[['model_name', 'f1_score', 'accuracy', 'roc_auc']])
```

### Best Model Selection
```python
# Get best model info
best_info = tracker.get_best_model_info(metric_name="f1_score")
print(f"Best model: {best_info['model_name']}")
print(f"F1 Score: {best_info['metrics']['f1_score']}")
```

## 🗂️ File Structure

```
project_files/mlflow/
├── __init__.py                    # Package initialization
├── mlflow_config.py               # Core MLflow configuration
├── experiment_tracker.py           # Experiment tracking utilities
├── run_experiments.py             # Experiment runner script
├── start_mlflow_ui.ps1           # MLflow UI launcher
├── README.md                      # This documentation
├── mlflow.db                      # SQLite tracking database
├── artifacts/                     # Model artifacts storage
└── confusion_matrix_*.png         # Generated confusion matrices
```

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
uv add mlflow matplotlib seaborn
```

### 2. Run Experiments
```powershell
uv run python project_files/mlflow/run_experiments.py
```

### 3. Start MLflow UI
```powershell
.\project_files\mlflow\start_mlflow_ui.ps1
```

### 4. View Results
- Open http://localhost:5000
- Browse experiments and runs
- Compare model performance
- Download artifacts

## 📋 Best Practices

1. **Always log parameters** for reproducibility
2. **Use consistent metric names** across experiments
3. **Export results** for external analysis
4. **Tag important runs** for easy identification
5. **Regular cleanup** of old artifacts

## 🔄 Integration

### With Existing Pipeline
```python
# In customer_churn_prediction.py
from project_files.mlflow import log_experiment_results

# After model training
log_experiment_results(
    params=model_params,
    metrics=model_metrics,
    model=trained_model
)
```

### With FastAPI
```python
# Load best model for API
from project_files.mlflow import load_best_model

best_model = load_best_model()
# Use in API predictions
```

## 📖 Documentation

- **MLflow Documentation**: https://mlflow.org/docs/
- **MLflow Python API**: https://mlflow.org/docs/latest/python_api/
- **MLflow UI Guide**: https://mlflow.org/docs/latest/tracking.html#tracking-ui

## 🛠️ Customization

### Adding New Metrics
```python
# In experiment_tracker.py
def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
    metrics = super()._calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Add custom metrics
    from sklearn.metrics import balanced_accuracy_score
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    return metrics
```

### Custom Model Logging
```python
# Log custom model types
def log_custom_model(self, model, model_name, custom_loader=None):
    if custom_loader:
        mlflow.pyfunc.log_model(model_name, python_model=custom_loader)
    else:
        mlflow.sklearn.log_model(model, model_name)
``` 