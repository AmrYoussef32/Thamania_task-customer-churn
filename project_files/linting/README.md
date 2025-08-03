# Code Quality and Linting Configuration

This folder contains all code quality, formatting, and linting configurations for the Customer Churn Prediction project.

## Files

### `ruff.toml`
- **Purpose**: Ruff configuration for linting and formatting
- **Features**:
  - Python 3.11 target
  - 88 character line length
  - Import sorting with isort
  - Flake8, Pycodestyle, Pyflakes integration
  - Mypy type checking
  - Per-file rule exclusions

### `pyproject.toml`
- **Purpose**: Black configuration for code formatting
- **Features**:
  - 88 character line length
  - Python 3.11 target
  - Exclude patterns for data and models

### `.pre-commit-config.yaml`
- **Purpose**: Pre-commit hooks for automated quality checks
- **Hooks**:
  - Ruff linting and formatting
  - Black formatting
  - Mypy type checking
  - Trailing whitespace removal
  - End-of-file fixer

### `lint.ps1`
- **Purpose**: PowerShell script for running all quality checks
- **Checks**:
  - Ruff linting
  - Ruff formatting check
  - Black formatting check
  - Mypy type checking

### `format.ps1`
- **Purpose**: PowerShell script for formatting code
- **Operations**:
  - Ruff formatting
  - Black formatting
  - Ruff auto-fix

## Usage

### From Project Root
```powershell
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

### From Linting Folder
```powershell
# Navigate to linting folder
cd project_files/linting

# Run checks
.\lint.ps1

# Format code
.\format.ps1
```

## Configuration Details

### Ruff Settings
- **Line Length**: 88 characters
- **Target Python**: 3.11
- **Quote Style**: Double quotes
- **Import Sorting**: Enabled with isort
- **Auto-fix**: Enabled for most issues

### Black Settings
- **Line Length**: 88 characters
- **Target Python**: 3.11
- **Exclude**: Models and data directories

### Mypy Settings
- **Python Version**: 3.11
- **Strict Mode**: Enabled
- **Type Checking**: Comprehensive

## 📋 Quality Standards

### Code Style
- ✅ **Consistent formatting** with Black and Ruff
- ✅ **Import organization** with isort
- ✅ **Line length** limited to 88 characters
- ✅ **Type hints** where appropriate
- ✅ **Docstrings** for functions and classes

### Linting Rules
- ✅ **Unused imports** detection and removal
- ✅ **Unused variables** detection
- ✅ **Code complexity** monitoring
- ✅ **Security issues** detection
- ✅ **Best practices** enforcement

### Exclusions
- `__init__.py` files: Unused import warnings
- `tests/` directory: Some strict rules relaxed
- `model_retraining/`: Complex functions allowed

## 🚀 Quick Commands

```powershell
# Check code quality
.\project_files\linting\lint.ps1

# Format code
.\project_files\linting\format.ps1

# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

## 📊 Benefits

1. **✅ Consistency**: Uniform code style across the project
2. **✅ Quality**: Automated detection of common issues
3. **✅ Maintainability**: Clean, readable code
4. **✅ Collaboration**: Standardized formatting for team
5. **✅ Automation**: Pre-commit hooks prevent bad commits

## 🔄 Integration

### With uv
```bash
# Install linting tools
uv add ruff black mypy pre-commit

# Run checks
uv run ruff check project_files/src/
```

### With Git Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run on staged files
pre-commit run
```

## 📖 Documentation

- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **Black Documentation**: https://black.readthedocs.io/
- **Mypy Documentation**: https://mypy.readthedocs.io/
- **Pre-commit Documentation**: https://pre-commit.com/ 