Write-Host "Syncing automation files..." -ForegroundColor Green

try {
    # Copy automation files from project_files/automation to root
    Write-Host "Copying automation files from project_files/automation to root..." -ForegroundColor Yellow
    
    Copy-Item "project_files/automation/Makefile" "./Makefile" -Force
    Copy-Item "project_files/automation/.pre-commit-config.yaml" "./.pre-commit-config.yaml" -Force
    
    Write-Host "Automation files synced to root directory" -ForegroundColor Green
    Write-Host "Remember: Edit source files in project_files/automation/ for best organization!" -ForegroundColor Cyan
} catch {
    Write-Host "Error syncing automation files" -ForegroundColor Red
    exit 1
} 