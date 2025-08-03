# Docker Setup for Customer Churn Prediction

This document explains how to use Docker for consistent deployment of the Customer Churn Prediction project.

## 🐳 Quick Start

### Prerequisites
- Docker installed
- Docker Compose installed
- At least 2GB RAM available

### 1. Build and Run with Docker Compose

```bash
# Build and start the API service
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop services
docker-compose down
```

### 2. Build and Run with Docker

```bash
# Build the image
docker build -t churn-prediction .

# Run the container
docker run -p 8000:8000 -v ./project_files/models:/app/project_files/models churn-prediction

# Run in background
docker run -d -p 8000:8000 -v ./project_files/models:/app/project_files/models --name churn-api churn-prediction
```

## 📁 Docker Structure

```
thamania_task_updated/
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Multi-service orchestration
├── .dockerignore                 # Files to exclude from build
├── DOCKER.md                     # This documentation
└── project_files/                # Application code
```

## 🔧 Services

### 1. **churn-api** (Main Service)
- **Purpose**: FastAPI application serving predictions
- **Port**: 8000
- **Health Check**: `/health` endpoint
- **Volumes**: 
  - `./project_files/models` → `/app/project_files/models`
  - `./project_files/data` → `/app/project_files/data`

### 2. **model-trainer** (Optional)
- **Purpose**: Train new models
- **Profile**: `training`
- **Usage**: `docker-compose --profile training up model-trainer`

## 🚀 Usage Examples

### Start API Service
```bash
# Start with logs
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build
```

### Train Model
```bash
# Train model using Docker
docker-compose --profile training run model-trainer

# Or build and run manually
docker build -t churn-prediction .
docker run -v ./project_files/models:/app/project_files/models \
           -v ./project_files/data:/app/project_files/data \
           churn-prediction \
           uv run python project_files/src/customer_churn_prediction.py
```

### Access the API
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
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

## 🔍 Troubleshooting

### Common Issues

#### 1. **Port Already in Use**
```bash
# Check what's using port 8000
netstat -tulpn | grep :8000

# Kill the process or change port in docker-compose.yml
```

#### 2. **Model Not Found**
```bash
# Ensure models directory exists
mkdir -p project_files/models

# Train model first
docker-compose --profile training run model-trainer
```

#### 3. **Build Fails**
```bash
# Clean build
docker-compose down
docker system prune -f
docker-compose up --build
```

#### 4. **Permission Issues**
```bash
# On Linux/Mac, ensure proper permissions
chmod -R 755 project_files/models
```

### Debug Commands

```bash
# View logs
docker-compose logs churn-api

# Enter container
docker-compose exec churn-api bash

# Check container status
docker-compose ps

# View resource usage
docker stats
```

## 📊 Performance

### Resource Requirements
- **Memory**: 1-2GB RAM
- **CPU**: 1-2 cores
- **Storage**: 500MB-1GB
- **Network**: 8000 port

### Optimization Tips
1. **Multi-stage builds** for smaller images
2. **Layer caching** for faster builds
3. **Volume mounting** for data persistence
4. **Health checks** for reliability

## 🔄 Development Workflow

### 1. **Local Development**
```bash
# Use local environment
uv sync
uv run python project_files/src/api/fastapi_app.py
```

### 2. **Docker Development**
```bash
# Build and run with Docker
docker-compose up --build
```

### 3. **Production Deployment**
```bash
# Build optimized image
docker build -t churn-prediction:prod .

# Run with production settings
docker run -d -p 8000:8000 \
  -v /path/to/models:/app/project_files/models \
  --restart unless-stopped \
  churn-prediction:prod
```

## 🛠️ Advanced Configuration

### Environment Variables
```yaml
# docker-compose.yml
environment:
  - PYTHONUNBUFFERED=1
  - HOST=0.0.0.0
  - PORT=8000
  - MODEL_PATH=/app/project_files/models
```

### Custom Dockerfile
```dockerfile
# Multi-stage build for smaller image
FROM python:3.11-slim as builder
# ... build dependencies

FROM python:3.11-slim as runtime
# ... runtime dependencies
```

## 📝 Best Practices

1. **Always use volumes** for data persistence
2. **Include health checks** for reliability
3. **Use .dockerignore** to exclude unnecessary files
4. **Tag images** with versions for reproducibility
5. **Monitor resource usage** in production
6. **Backup models** regularly
7. **Use restart policies** for resilience

## 🎯 Benefits of Docker

- ✅ **Consistency**: Same environment everywhere
- ✅ **Isolation**: No conflicts with system packages
- ✅ **Portability**: Run on any Docker-enabled system
- ✅ **Scalability**: Easy to deploy multiple instances
- ✅ **Versioning**: Reproducible builds
- ✅ **CI/CD**: Easy integration with pipelines 