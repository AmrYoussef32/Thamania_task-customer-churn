# Makefile for Customer Churn Prediction Project
# Automated tasks for development and deployment

.PHONY: help install test lint format clean build run-api train-model docker-build docker-run sync-all

# Default target
help:
	@echo "🤖 Customer Churn Prediction - Automated Tasks"
	@echo "=============================================="
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  install        Install all dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  test           Run all tests"
	@echo "  lint           Run code quality checks"
	@echo "  format         Format code with Black and Ruff"
	@echo "  quality        Run lint + format + test"
	@echo ""
	@echo "🚀 Development:"
	@echo "  run-api        Start FastAPI server"
	@echo "  train-model    Train the ML model"
	@echo "  monitor        Run model monitoring"
	@echo "  retrain        Run model retraining"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run with Docker Compose"
	@echo "  docker-stop    Stop Docker services"
	@echo ""
	@echo "🔬 MLflow:"
	@echo "  mlflow         Run MLflow experiments"
	@echo "  mlflow-ui      Start MLflow UI"
	@echo ""
	@echo "🔍 Monitoring:"
	@echo "  monitoring     Run monitoring demo"
	@echo "  monitor-sched  Run scheduled monitoring"
	@echo ""
	@echo "🔄 Sync & Maintenance:"
	@echo "  sync-all       Sync all configuration files"
	@echo "  clean          Clean temporary files"
	@echo "  build          Build project artifacts"
	@echo ""

# Installation
install:
	@echo "📦 Installing dependencies..."
	uv sync

install-dev:
	@echo "📦 Installing development dependencies..."
	uv sync --dev

# Testing and Quality
test:
	@echo "🧪 Running tests..."
	uv run pytest

lint:
	@echo "🔍 Running code quality checks..."
	uv run ruff check project_files/src/
	uv run black --check project_files/src/

format:
	@echo "🎨 Formatting code..."
	uv run ruff format project_files/src/
	uv run black project_files/src/
	uv run ruff check --fix project_files/src/

quality: lint format test
	@echo "✅ All quality checks completed!"

# Development
run-api:
	@echo "🚀 Starting FastAPI server..."
	uv run python project_files/src/api/fastapi_app.py

train-model:
	@echo "🤖 Training ML model..."
	uv run python project_files/src/customer_churn_prediction.py

monitor:
	@echo "📊 Running model monitoring..."
	uv run python project_files/src/model_retraining/model_monitoring.py

retrain:
	@echo "🔄 Running model retraining..."
	uv run python project_files/src/model_retraining/model_retraining.py

# Docker
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t churn-prediction .

docker-run:
	@echo "🐳 Starting Docker services..."
	docker-compose up --build

docker-stop:
	@echo "🐳 Stopping Docker services..."
	docker-compose down

# Sync and Maintenance
sync-all:
	@echo "🔄 Syncing all configuration files..."
	./sync_config.ps1
	./sync_docker.ps1
	./sync_linting.ps1

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

build: clean install-dev quality
	@echo "🏗️ Building project artifacts..."
	@echo "✅ Build completed successfully!"

# Pre-commit hooks
install-hooks:
	@echo "🔗 Installing pre-commit hooks..."
	uv run pre-commit install

run-hooks:
	@echo "🔗 Running pre-commit hooks..."
	uv run pre-commit run --all-files

# Quick development workflow
dev: install-dev quality run-api
	@echo "🚀 Development environment ready!"

# Production deployment
prod: install docker-build docker-run
	@echo "🚀 Production deployment ready!"

# Model pipeline
pipeline: train-model monitor retrain
	@echo "🔄 Model pipeline completed!"

# MLflow experiments
mlflow:
	@echo "🔬 Running MLflow experiments..."
	uv run python project_files/mlflow/run_experiments.py

mlflow-ui:
	@echo "🔬 Starting MLflow UI..."
	./project_files/mlflow/start_mlflow_ui.ps1

# Monitoring
monitoring:
	@echo "🔍 Running monitoring demo..."
	uv run python project_files/monitoring/run_monitoring.py --mode demo

monitor-sched:
	@echo "🕐 Running scheduled monitoring..."
	uv run python project_files/monitoring/run_monitoring.py --mode scheduled 