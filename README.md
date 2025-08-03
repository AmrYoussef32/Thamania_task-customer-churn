# Customer Churn Prediction Project

A comprehensive machine learning project for predicting customer churn using FastAPI and automated retraining.

## Quick Start

### Using Docker (Recommended for Production)

```bash
# Check Docker setup
.\project_files\docker\docker-setup.ps1

# Build and run with Docker Compose
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### Using uv (Recommended for Development)

```bash
# Install dependencies
uv sync

# Run the ML model
uv run python project_files/src/customer_churn_prediction.py

# Start the FastAPI server
uv run python project_files/src/api/fastapi_app.py
```

### Using pip

```bash
# Install dependencies
pip install -r project_files/requirements.txt

# Run the ML model
python project_files/src/customer_churn_prediction.py

# Start the FastAPI server
cd project_files/src/api
python fastapi_app.py
```

## Project Structure

```
thamania_task_updated/
├── project_config/                     # Project configuration files
│   ├── pyproject.toml                  # uv project configuration (source)
│   ├── .python-version                 # Python version specification
│   └── README.md                       # Configuration documentation
├── project_files/                      # Main project code
│   ├── src/
│   │   ├── api/
│   │   │   └── fastapi_app.py          # FastAPI application
│   │   ├── customer_churn_prediction.py # ML model training
│   │   └── model_retraining/           # Automated retraining system
│   ├── data/
│   │   └── customer_churn_mini.json    # Dataset
│   ├── models/                         # Trained models
│   ├── config/
│   │   └── retraining_config.json      # Retraining configuration
│   ├── docker/                         # Docker configuration
│   │   ├── Dockerfile                  # Docker image definition
│   │   ├── docker-compose.yml          # Multi-service orchestration
│   │   ├── .dockerignore               # Docker build exclusions
│   │   ├── DOCKER.md                   # Docker documentation
│   │   ├── docker-setup.ps1            # Docker setup script
│   │   └── README.md                   # Docker documentation
│   ├── linting/                        # Code quality and linting
│   │   ├── ruff.toml                   # Ruff configuration
│   │   ├── pyproject.toml              # Black configuration
│   │   ├── .pre-commit-config.yaml     # Pre-commit hooks
│   │   ├── lint.ps1                    # Linting script
│   │   ├── format.ps1                  # Formatting script
│   │   └── README.md                   # Linting documentation
│   ├── automation/                      # Automation and task management
│   │   ├── Makefile                    # Unix/Linux task automation
│   │   ├── tasks.ps1                   # PowerShell task automation
│   │   ├── .pre-commit-config.yaml     # Enhanced pre-commit hooks
│   │   ├── setup-automation.ps1        # Automation setup script
│   │   └── README.md                   # Automation documentation
│   ├── mlflow/                         # MLflow experiment tracking
│   │   ├── mlflow_config.py            # MLflow configuration
│   │   ├── experiment_tracker.py       # Experiment tracking utilities
│   │   ├── run_experiments.py          # Experiment runner script
│   │   ├── start_mlflow_ui.ps1        # MLflow UI launcher
│   │   └── README.md                   # MLflow documentation
│   ├── monitoring/                     # Monitoring and drift detection
│   │   ├── drift_detection.py          # Data and concept drift detection
│   │   ├── performance_monitor.py      # Performance tracking and analysis
│   │   ├── monitoring_orchestrator.py  # Unified monitoring coordination
│   │   ├── run_monitoring.py           # Monitoring runner and demo
│   │   └── README.md                   # Monitoring documentation
│   ├── documentation/                  # Professional documentation
│   │   ├── technical_report.md         # Comprehensive technical report
│   │   ├── README.md                   # Solution approach and challenges
│   │   └── api_documentation.md        # API usage and examples
│   ├── requirements.txt                # pip dependencies
│   └── README_api.md                   # API documentation
├── Dockerfile                          # Docker image definition (copy)
├── docker-compose.yml                  # Multi-service orchestration (copy)
├── .dockerignore                       # Docker build exclusions (copy)
├── pyproject.toml                      # uv project configuration (copy)
├── uv.lock                             # uv dependency lock file
├── sync_config.ps1                     # Configuration sync script
├── sync_docker.ps1                     # Docker sync script
├── sync_linting.ps1                    # Linting sync script
├── sync_automation.ps1                 # Automation sync script
└── README.md                          # This file
```

## Features

### Machine Learning
- ✅ Customer churn prediction using scikit-learn
- ✅ Data leakage prevention with temporal splitting
- ✅ Feature engineering and preprocessing
- ✅ Model evaluation and selection
- ✅ Model persistence and versioning

### FastAPI Service
- ✅ RESTful API for predictions
- ✅ Interactive documentation at `/docs`
- ✅ Health checks and monitoring
- ✅ Batch prediction support
- ✅ Human-friendly responses with business recommendations

### Automated Retraining
- ✅ Periodic model retraining
- ✅ Performance monitoring
- ✅ Data drift detection
- ✅ Automatic model backup and replacement
- ✅ Configurable retraining criteria

### Docker Containerization
- ✅ Consistent runtime environment
- ✅ Easy deployment and scaling
- ✅ Volume mounting for data persistence
- ✅ Health checks and monitoring
- ✅ Multi-service orchestration

### Automation and Task Management
- ✅ Makefile for Unix/Linux automation
- ✅ PowerShell tasks for Windows automation
- ✅ Pre-commit hooks for quality assurance
- ✅ One-command workflows (dev, prod, pipeline)
- ✅ Cross-platform task management

### MLflow Experiment Tracking
- ✅ Experiment tracking and versioning
- ✅ Model comparison and selection
- ✅ Comprehensive metrics logging
- ✅ MLflow UI for visualization
- ✅ Best model retrieval and loading

### Monitoring and Drift Detection
- ✅ Data drift detection using statistical tests
- ✅ Concept drift detection based on performance
- ✅ Ongoing performance monitoring and tracking
- ✅ Comprehensive drift analysis and reporting
- ✅ Actionable recommendations for model maintenance

## Development

### Install Development Dependencies

```bash
# Using uv
uv sync --dev

# Using pip
pip install -r project_files/requirements.txt
```

### Run Tests

```bash
# Using uv
uv run pytest

# Using pip
pytest
```

### Code Quality and Formatting

```bash
# Run all quality checks
.\project_files\linting\lint.ps1

# Format code
.\project_files\linting\format.ps1

# Run individual tools
uv run ruff check project_files/src/
uv run ruff format project_files/src/
uv run black project_files/src/
uv run mypy project_files/src/
```

### Automation and Task Management

```bash
# Setup automation tools
.\project_files\automation\setup-automation.ps1

# Show all available tasks
.\project_files\automation\tasks.ps1 help

# Development workflow
.\project_files\automation\tasks.ps1 dev

# Quality checks
.\project_files\automation\tasks.ps1 quality

# Production deployment
.\project_files\automation\tasks.ps1 prod
```

### MLflow Experiment Tracking

```bash
# Run MLflow experiments
uv run python project_files/mlflow/run_experiments.py

# Start MLflow UI
.\project_files\mlflow\start_mlflow_ui.ps1

# Access MLflow UI at http://localhost:5000
```

### Monitoring and Drift Detection

```bash
# Run monitoring demo
uv run python project_files/monitoring/run_monitoring.py --mode demo

# Run scheduled monitoring
uv run python project_files/monitoring/run_monitoring.py --mode scheduled
```

## 📚 Documentation

### Technical Documentation
- **Technical Report**: `project_files/documentation/technical_report.md` - Comprehensive technical report covering data preparation, modeling steps, features, model choices, retraining strategy, technical challenges, and improvement suggestions
- **Solution Approach**: `project_files/documentation/README.md` - Detailed documentation of solution approach, main difficulties encountered, and suggestions for alternative/improved solutions
- **API Documentation**: `project_files/documentation/api_documentation.md` - Complete API documentation with usage examples and endpoint descriptions

### Quick Documentation Access
```bash
# View technical report
cat project_files/documentation/technical_report.md

# View solution approach
cat project_files/documentation/README.md

# View API documentation
cat project_files/documentation/api_documentation.md
```

## API Usage

### Start the API Server

```bash
cd project_files/src/api
python fastapi_app.py
```

### Access the API

- **Interactive Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Predictions**: http://localhost:8000/predict

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "customer_001",
    "total_sessions": 15,
    "total_events": 150,
    "page_diversity": 8,
    "artist_diversity": 25,
    "song_diversity": 45,
    "total_length": 3600.5,
    "avg_song_length": 240.0,
    "days_active": 30,
    "events_per_session": 10.0,
    "level": "paid",
    "gender": "M",
    "registration": 1538352000000
  }'
```

## Model Retraining

### Manual Retraining

```bash
python project_files/src/model_retraining/model_retraining.py
```

### Automated Retraining

```bash
# Weekly retraining
python project_files/src/model_retraining/retraining_scheduler.py weekly

# Performance monitoring
python project_files/src/model_retraining/retraining_scheduler.py monitor
```

## Performance

- **Model Accuracy**: F1 Score ~0.778
- **API Response Time**: ~50ms per prediction
- **Throughput**: ~1000 requests/second

## Technologies Used

- **Python 3.8+**
- **FastAPI** - Web framework
- **scikit-learn** - Machine learning
- **pandas** - Data manipulation
- **uv** - Package management
- **pytest** - Testing

## License

This project is for educational and demonstration purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run pytest`
5. Submit a pull request
