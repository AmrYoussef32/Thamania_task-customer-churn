Write-Host "Syncing linting configuration files..." -ForegroundColor Green

try {
    # Copy linting files from project_files/linting to root
    Write-Host "Copying linting files from project_files/linting to root..." -ForegroundColor Yellow
    
    Copy-Item "project_files/linting/ruff.toml" "./ruff.toml" -Force
    Copy-Item "project_files/linting/pyproject.toml" "./pyproject.toml" -Force
    Copy-Item "project_files/linting/.pre-commit-config.yaml" "./.pre-commit-config.yaml" -Force
    
    Write-Host "Linting files synced to root directory" -ForegroundColor Green
    Write-Host "Remember: Edit source files in project_files/linting/ for best organization!" -ForegroundColor Cyan
} catch {
    Write-Host "Error syncing linting files" -ForegroundColor Red
    exit 1
} 