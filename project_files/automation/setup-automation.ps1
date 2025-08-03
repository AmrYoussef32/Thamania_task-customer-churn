#!/usr/bin/env pwsh

Write-Host "🤖 Setting up Automation Tools..." -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Change to project root
Set-Location $PSScriptRoot\..\..

try {
    # Install additional automation dependencies
    Write-Host "📦 Installing automation dependencies..." -ForegroundColor Yellow
    uv add --dev bandit pydocstyle isort yamllint

    # Install pre-commit hooks
    Write-Host "🔗 Installing pre-commit hooks..." -ForegroundColor Yellow
    uv run pre-commit install

    # Copy automation files to root
    Write-Host "📁 Setting up automation files..." -ForegroundColor Yellow
    Copy-Item "project_files/automation/Makefile" "./Makefile" -Force
    Copy-Item "project_files/automation/.pre-commit-config.yaml" "./.pre-commit-config.yaml" -Force

    # Test automation setup
    Write-Host "🧪 Testing automation setup..." -ForegroundColor Yellow
    uv run pre-commit run --all-files

    Write-Host "✅ Automation setup completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 Available automation commands:" -ForegroundColor Cyan
    Write-Host "  make help                    # Show all available tasks" -ForegroundColor Gray
    Write-Host "  make install-dev             # Install development dependencies" -ForegroundColor Gray
    Write-Host "  make quality                 # Run all quality checks" -ForegroundColor Gray
    Write-Host "  make dev                     # Setup development environment" -ForegroundColor Gray
    Write-Host "  make prod                    # Setup production environment" -ForegroundColor Gray
    Write-Host "  .\tasks.ps1 help             # Show PowerShell tasks" -ForegroundColor Gray
    Write-Host "  .\tasks.ps1 quality          # Run quality checks" -ForegroundColor Gray
    Write-Host "  .\tasks.ps1 dev              # Setup development environment" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🔗 Pre-commit hooks are now active!" -ForegroundColor Green
    Write-Host "   They will run automatically on git commit" -ForegroundColor Cyan

} catch {
    Write-Host "❌ Automation setup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 