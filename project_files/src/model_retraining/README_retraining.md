# Model Retraining System

Simple automated retraining system for the customer churn prediction model.

## 📁 File Structure

```
project_files/
├── config/
│   └── retraining_config.json    # Configuration settings
├── src/
│   ├── customer_churn_prediction.py  # Main prediction model
│   └── model_retraining/
│       ├── __init__.py              # Package initialization
│       ├── model_monitoring.py      # Monitor model performance
│       ├── model_retraining.py      # Retrain model
│       └── retraining_scheduler.py  # Scheduler for automation
└── models/
    ├── backup/                   # Model backups
    └── retraining_log.txt        # Retraining logs
```

## 🚀 Quick Start

### 1. Check if retraining is needed:
```bash
python project_files/src/model_retraining/model_monitoring.py
```

### 2. Retrain model manually:
```bash
python project_files/src/model_retraining/model_retraining.py
```

### 3. Force retraining (ignore performance check):
```bash
python project_files/src/model_retraining/model_retraining.py --force
```

### 4. Run scheduled check:
```bash
python project_files/src/model_retraining/retraining_scheduler.py check
```

## ⚙️ Configuration

Edit `project_files/config/retraining_config.json`:

```json
{
  "retraining_schedule": "weekly",
  "performance_thresholds": {
    "f1_min": 0.7,
    "auc_min": 0.6,
    "precision_min": 0.5,
    "recall_min": 0.3
  },
  "drift_threshold": 0.2,
  "min_improvement": 0.02
}
```

## 📅 Scheduling

### Windows Task Scheduler:
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., Weekly on Sunday at 2 AM)
4. Action: Start a program
5. Program: `python`
6. Arguments: `project_files/src/model_retraining/retraining_scheduler.py check`

### Linux Cron:
```bash
# Check daily at 2 AM
0 2 * * * cd /path/to/project && python project_files/src/model_retraining/retraining_scheduler.py check

# Force weekly retraining on Sundays
0 3 * * 0 cd /path/to/project && python project_files/src/model_retraining/retraining_scheduler.py weekly
```

## 🔍 Monitoring

### Performance Monitoring:
- F1 Score threshold: 0.7
- AUC Score threshold: 0.6
- Precision threshold: 0.5
- Recall threshold: 0.3

### Data Drift Detection:
- Numerical features: 20% change in mean
- Categorical features: Chi-square test (p < 0.05)

## 📊 Logging

All retraining events are logged to `project_files/models/retraining_log.txt`:

```
[2025-08-02 21:30:00] RETRAINING_STARTED: Manual retraining triggered
[2025-08-02 21:35:00] MODEL_REPLACED: New model better: F1 +0.050, AUC +0.030
[2025-08-02 21:40:00] FORCE_RETRAINING: Force retraining triggered
```

## 🛡️ Safety Features

### Model Backup:
- Current model is backed up before replacement
- Backups stored in `project_files/models/backup/`

### Performance Comparison:
- New model must be better by at least 2% (configurable)
- Only replaces if F1 or AUC improves significantly

### Error Handling:
- Graceful failure handling
- Detailed error logging
- Rollback capability

## 🔧 Commands

| Command | Description |
|---------|-------------|
| `python model_retraining/model_monitoring.py` | Check if retraining needed |
| `python model_retraining/model_retraining.py` | Retrain if needed |
| `python model_retraining/model_retraining.py --force` | Force retraining |
| `python model_retraining/retraining_scheduler.py check` | Scheduled check |
| `python model_retraining/retraining_scheduler.py weekly` | Weekly retraining |
| `python model_retraining/retraining_scheduler.py force` | Force retraining |

## 📈 Metrics

The system monitors:
- **F1 Score**: Overall model performance
- **AUC Score**: Model discrimination ability
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positives
- **Data Drift**: Feature distribution changes

## 🎯 Retraining Triggers

1. **Performance Drop**: Any metric below threshold
2. **Data Drift**: Feature distributions change significantly
3. **Time-based**: Weekly/monthly scheduled retraining
4. **Manual**: Force retraining when needed

## 🔄 Workflow

1. **Monitor**: Check model performance daily
2. **Detect**: Identify when retraining is needed
3. **Backup**: Save current model
4. **Train**: Create new model with latest data
5. **Compare**: Test new vs current model
6. **Deploy**: Replace only if better
7. **Log**: Record all activities

## ⚠️ Troubleshooting

### Common Issues:
- **No model found**: Run initial training first
- **Config file missing**: System uses default settings
- **Data path error**: Check data file location
- **Permission error**: Ensure write access to models directory

### Debug Mode:
Add `--debug` flag to any command for detailed output:
```bash
python project_files/src/model_retraining/model_monitoring.py --debug
``` 