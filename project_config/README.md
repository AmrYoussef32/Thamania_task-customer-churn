# Project Configuration

This folder contains all configuration files for the Customer Churn Prediction project.

## 📁 Files

### `pyproject.toml`
- **Purpose**: Main project configuration for `uv` package management
- **Contains**: 
  - Project metadata (name, version, description)
  - Python dependencies
  - Development dependencies
  - uv-specific settings
- **Usage**: `uv sync` reads this file to install dependencies

### `.python-version`
- **Purpose**: Specifies the Python version for the project
- **Created by**: `uv init`
- **Usage**: Ensures consistent Python version across environments

## 🔧 How to Use

### Install Dependencies
```bash
# From project root
uv sync
```

### Add New Dependencies
```bash
# Add production dependency
uv add package-name

# Add development dependency
uv add --dev package-name
```

### Update Dependencies
```bash
# Update all dependencies
uv sync --upgrade
```

## 📋 Configuration Management

The `pyproject.toml` in the root directory is a copy of the one in `project_config/`. 
When making changes:

1. **Edit**: `project_config/pyproject.toml` (source of truth)
2. **Sync**: Use the sync script: `.\sync_config.ps1`
3. **Commit**: Both files to version control

### 🔄 Sync Script Usage

```powershell
# Sync from project_config to root (recommended)
.\sync_config.ps1

# Sync from root to project_config (if needed)
.\sync_config.ps1 -Direction "from-root"
```

## 🎯 Benefits

- **Organization**: All config files in one place
- **Clarity**: Clear separation of code and configuration
- **Maintainability**: Easy to find and update settings
- **Documentation**: Self-documenting structure 