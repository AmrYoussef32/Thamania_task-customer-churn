# Docker Configuration

This folder contains all Docker-related files for the Customer Churn Prediction project.

## Files

### `Dockerfile`
- **Purpose**: Docker image definition for the application
- **Base Image**: Python 3.11 slim
- **Features**: 
  - uv package management
  - FastAPI application
  - Volume mounting for models
  - Health checks

### `docker-compose.yml`
- **Purpose**: Multi-service orchestration
- **Services**:
  - `churn-api`: Main FastAPI service
  - `model-trainer`: Optional training service
- **Features**:
  - Volume persistence
  - Health monitoring
  - Easy scaling

### `.dockerignore`
- **Purpose**: Exclude unnecessary files from Docker build
- **Excludes**:
  - Git files
  - Python cache
  - Virtual environments
  - IDE files
  - OS files
  - Model files (for faster builds)

### `DOCKER.md`
- **Purpose**: Comprehensive Docker documentation
- **Contains**:
  - Quick start guide
  - Usage examples
  - Troubleshooting
  - Best practices
  - Performance tips

### `docker-setup.ps1`
- **Purpose**: PowerShell script for Docker setup verification
- **Features**:
  - Docker installation check
  - Docker runtime check
  - Command reference
  - Setup guidance

## Usage

### From Project Root
```bash
# Build and run
docker-compose up --build

# Train model
docker-compose --profile training run model-trainer

# Check setup
.\project_files\docker\docker-setup.ps1
```

### From Docker Folder
```bash
# Navigate to docker folder
cd project_files/docker

# Build image
docker build -t churn-prediction ..

# Run container
docker run -p 8000:8000 churn-prediction
```

## Organization Benefits

1. **✅ Centralized**: All Docker files in one place
2. **✅ Organized**: Clear separation of concerns
3. **✅ Maintainable**: Easy to find and update
4. **✅ Documented**: Self-documenting structure
5. **✅ Compatible**: Root copies for Docker commands

## File Management

The Docker files exist in two locations:
- **Source**: `project_files/docker/` (for organization)
- **Root**: `./` (for Docker compatibility)

When making changes:
1. **Edit**: Files in `project_files/docker/`
2. **Copy**: To root directory for Docker to find
3. **Commit**: Both locations to version control

## Docker Architecture

```
project_files/docker/
├── Dockerfile              # Image definition
├── docker-compose.yml      # Service orchestration
├── .dockerignore           # Build exclusions
├── DOCKER.md              # Documentation
├── docker-setup.ps1       # Setup script
└── README.md              # This file
```

## 🚀 Quick Commands

```bash
# Start API
docker-compose up --build

# Train model
docker-compose --profile training run model-trainer

# View logs
docker-compose logs churn-api

# Stop services
docker-compose down
```

## 📖 Documentation

- **Setup Guide**: `DOCKER.md`
- **API Documentation**: `../README_api.md`
- **Project Overview**: `../../README.md` 