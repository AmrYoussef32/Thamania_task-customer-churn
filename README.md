# Customer Churn Prediction Project

Hey there! This is a comprehensive machine learning project that predicts customer churn using FastAPI and automated retraining. Think of it as your AI-powered crystal ball for understanding when customers might leave your service.

## Quick Start

### Option 1: Docker (Production Ready)
If you want to get up and running quickly with everything containerized:

```bash
# First, let's make sure Docker is set up properly
.\project_files\docker\docker-setup.ps1

# Now let's build and run everything
docker-compose up --build

# Or run it in the background if you prefer
docker-compose up -d --build
```

### Option 2: uv (Development Friendly)
For developers who love fast package management:

```bash
# Install all the dependencies
uv sync

# Train the ML model
uv run python project_files/src/customer_churn_prediction.py

# Start the API server
uv run python project_files/src/api/fastapi_app.py
```

### Option 3: Traditional pip
The classic approach that works everywhere:

```bash
# Install dependencies
pip install -r project_files/requirements.txt

# Run the ML model
python project_files/src/customer_churn_prediction.py

# Start the FastAPI server
cd project_files/src/api
python fastapi_app.py
```

## What's Inside

Here's how we've organized this project to keep things clean and maintainable:

```
thamania_task_updated/
├── project_config/                     # Configuration files
│   ├── pyproject.toml                  # uv project configuration
│   ├── .python-version                 # Python version specification
│   └── README.md                       # Configuration docs
├── project_files/                      # The main event - all our code!
│   ├── src/
│   │   ├── api/
│   │   │   └── fastapi_app.py          # Our FastAPI application
│   │   ├── customer_churn_prediction.py # The ML model training
│   │   └── model_retraining/           # Automated retraining system
│   ├── data/
│   │   └── customer_churn_mini.json    # Our dataset
│   ├── models/                         # Trained models live here
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

## What Makes This Project Special

### Machine Learning Magic
- **Smart Churn Prediction**: Uses scikit-learn to predict when customers might leave
- **Data Leakage Prevention**: Implements temporal splitting to avoid cheating
- **Feature Engineering**: Creates meaningful features from raw data
- **Model Evaluation**: Comprehensive testing and selection process
- **Model Persistence**: Saves and versions models for production use

### FastAPI Service
- **RESTful API**: Clean endpoints for predictions
- **Interactive Docs**: Beautiful documentation at `/docs`
- **Health Checks**: Monitor your service health
- **Batch Predictions**: Handle multiple predictions efficiently
- **Business-Friendly Responses**: Get actionable insights, not just numbers

### Automated Retraining
- **Periodic Updates**: Models retrain automatically
- **Performance Monitoring**: Track how well your models are doing
- **Data Drift Detection**: Know when your data changes
- **Automatic Backups**: Safe model replacement
- **Configurable Criteria**: Set your own retraining rules

### Docker Containerization
- **Consistent Environment**: Same setup everywhere
- **Easy Deployment**: Deploy and scale effortlessly
- **Data Persistence**: Keep your data safe
- **Health Monitoring**: Built-in health checks
- **Multi-Service**: Orchestrate multiple components

### Automation and Task Management
- **Makefile**: Unix/Linux automation
- **PowerShell Tasks**: Windows automation
- **Pre-commit Hooks**: Quality assurance
- **One-command Workflows**: Simplify your development
- **Cross-platform**: Works everywhere

### MLflow Experiment Tracking
- **Experiment Tracking**: Keep track of all your experiments
- **Model Comparison**: Compare different models easily
- **Metrics Logging**: Comprehensive performance tracking
- **MLflow UI**: Beautiful visualization interface
- **Best Model Retrieval**: Always get the best model

### Monitoring and Drift Detection
- **Data Drift Detection**: Statistical tests to detect changes
- **Concept Drift Detection**: Performance-based drift detection
- **Ongoing Monitoring**: Continuous performance tracking
- **Comprehensive Analysis**: Detailed drift reports
- **Actionable Recommendations**: Get suggestions for model maintenance

## Development

### Installing Development Dependencies

```bash
# Using uv (recommended)
uv sync --dev

# Using pip
pip install -r project_files/requirements.txt
```

### Running Tests

```bash
# Using uv
uv run pytest

# Using pip
pytest
```

### Code Quality and Formatting

```bash
# Run all quality checks at once
.\project_files\linting\lint.ps1

# Format your code
.\project_files\linting\format.ps1

# Or run individual tools
uv run ruff check project_files/src/
uv run ruff format project_files/src/
uv run black project_files/src/
uv run mypy project_files/src/
```

### Automation and Task Management

```bash
# Set up automation tools
.\project_files\automation\setup-automation.ps1

# See all available tasks
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

## Documentation

### Technical Documentation
- **Technical Report**: `project_files/documentation/technical_report.md` - Deep dive into data preparation, modeling steps, features, model choices, retraining strategy, technical challenges, and improvement suggestions
- **Solution Approach**: `project_files/documentation/README.md` - Detailed walkthrough of our solution approach, main difficulties we encountered, and suggestions for alternative/improved solutions
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

### Starting the API Server

```bash
cd project_files/src/api
python fastapi_app.py
```

### Accessing the API

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

- **Python 3.8+** - Our programming language of choice
- **FastAPI** - Modern web framework for APIs
- **scikit-learn** - Machine learning powerhouse
- **pandas** - Data manipulation wizard
- **uv** - Lightning-fast package management
- **pytest** - Testing framework

## License

This project is for educational and demonstration purposes.

## Contributing

We'd love to have you contribute! Here's how:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run pytest`
5. Submit a pull request

---


