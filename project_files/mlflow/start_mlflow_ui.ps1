#!/usr/bin/env pwsh

Write-Host "🔬 Starting MLflow UI..." -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green

try {
    # Change to project root
    Set-Location $PSScriptRoot\..\..
    
    # Check if MLflow is installed
    Write-Host "📦 Checking MLflow installation..." -ForegroundColor Yellow
    uv run python -c "import mlflow; print('MLflow is available')"
    
    # Start MLflow UI
    Write-Host "🚀 Starting MLflow UI server..." -ForegroundColor Yellow
    Write-Host "📊 MLflow UI will be available at: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "📁 Tracking URI: sqlite:///project_files/mlflow/mlflow.db" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    
    # Start MLflow UI
    uv run mlflow ui --backend-store-uri sqlite:///project_files/mlflow/mlflow.db --port 5000
    
} catch {
    Write-Host "❌ Error starting MLflow UI: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Make sure MLflow is installed:" -ForegroundColor Yellow
    Write-Host "   uv add mlflow" -ForegroundColor Gray
    exit 1
} 