# Automation and Task Management

This folder contains all automation tools, task management, and pre-commit hooks for the Customer Churn Prediction project.

## 📁 Files

### `Makefile`
- **Purpose**: Unix/Linux task automation with make
- **Features**:
  - Installation tasks (install, install-dev)
  - Quality checks (test, lint, format, quality)
  - Development tasks (run-api, train-model, monitor, retrain)
  - Docker tasks (docker-build, docker-run, docker-stop)
  - Maintenance tasks (sync-all, clean, build)
  - Pre-commit hooks (install-hooks, run-hooks)
  - Workflow tasks (dev, prod, pipeline)

### `tasks.ps1`
- **Purpose**: PowerShell equivalent of Makefile for Windows
- **Features**:
  - Same functionality as Makefile
  - Windows-compatible commands
  - Error handling and colored output
  - Parameter-based task execution

### `.pre-commit-config.yaml`
- **Purpose**: Enhanced pre-commit hooks configuration
- **Hooks**:
  - **Ruff**: Linting and formatting
  - **Black**: Code formatting
  - **Mypy**: Type checking
  - **isort**: Import sorting
  - **bandit**: Security checks
  - **pydocstyle**: Documentation checks
  - **File hygiene**: Trailing whitespace, end-of-file
  - **Custom checks**: Project structure, sync scripts

### `setup-automation.ps1`
- **Purpose**: PowerShell script to set up automation tools
- **Features**:
  - Install additional dependencies
  - Install pre-commit hooks
  - Copy automation files to root
  - Test automation setup

## 🔧 Usage

### From Project Root (Unix/Linux)
```bash
# Show all available tasks
make help

# Install dependencies
make install-dev

# Run quality checks
make quality

# Development workflow
make dev

# Production deployment
make prod
```

### From Project Root (Windows)
```powershell
# Show all available tasks
.\tasks.ps1 help

# Install dependencies
.\tasks.ps1 install-dev

# Run quality checks
.\tasks.ps1 quality

# Development workflow
.\tasks.ps1 dev

# Production deployment
.\tasks.ps1 prod
```

### From Automation Folder
```powershell
# Navigate to automation folder
cd project_files/automation

# Setup automation tools
.\setup-automation.ps1

# Run tasks from automation folder
.\tasks.ps1 help
```

## 🎯 Available Tasks

### Setup & Installation
- **install**: Install all dependencies
- **install-dev**: Install development dependencies

### Testing & Quality
- **test**: Run all tests
- **lint**: Run code quality checks
- **format**: Format code with Black and Ruff
- **quality**: Run lint + format + test

### Development
- **run-api**: Start FastAPI server
- **train-model**: Train the ML model
- **monitor**: Run model monitoring
- **retrain**: Run model retraining

### Docker
- **docker-build**: Build Docker image
- **docker-run**: Run with Docker Compose
- **docker-stop**: Stop Docker services

### Sync & Maintenance
- **sync-all**: Sync all configuration files
- **clean**: Clean temporary files
- **build**: Build project artifacts

### Pre-commit
- **install-hooks**: Install pre-commit hooks
- **run-hooks**: Run pre-commit hooks

### Workflows
- **dev**: Development environment setup
- **prod**: Production deployment
- **pipeline**: Model training pipeline

## 🔗 Pre-commit Hooks

### Automatic Hooks
The following hooks run automatically on `git commit`:

1. **Ruff linting and auto-fix**
2. **Ruff formatting**
3. **Black formatting**
4. **Mypy type checking**
5. **Import sorting (isort)**
6. **Security checks (bandit)**
7. **Documentation checks (pydocstyle)**
8. **File hygiene (trailing whitespace, end-of-file)**
9. **Custom project checks**

### Manual Hook Execution
```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files

# Run hooks on staged files only
uv run pre-commit run
```

## 🚀 Quick Start

### 1. Setup Automation
```powershell
# Setup automation tools
.\project_files\automation\setup-automation.ps1
```

### 2. Development Workflow
```powershell
# Complete development setup
.\tasks.ps1 dev
```

### 3. Quality Assurance
```powershell
# Run all quality checks
.\tasks.ps1 quality
```

### 4. Production Deployment
```powershell
# Complete production setup
.\tasks.ps1 prod
```

## 📊 Benefits

1. **✅ Automation**: Reduce manual tasks and errors
2. **✅ Consistency**: Standardized workflows across team
3. **✅ Quality**: Automated checks prevent issues
4. **✅ Efficiency**: One-command workflows
5. **✅ Cross-platform**: Works on Unix/Linux and Windows

## 🔄 Integration

### With Git
```bash
# Pre-commit hooks run automatically
git add .
git commit -m "Your commit message"
# Hooks run automatically before commit
```

### With CI/CD
```bash
# Use Makefile in CI/CD pipelines
make quality
make test
make build
```

### With Development
```bash
# Quick development setup
make dev
# or
.\tasks.ps1 dev
```

## 📋 Best Practices

1. **Always run quality checks** before committing
2. **Use pre-commit hooks** to catch issues early
3. **Run full pipeline** before deployment
4. **Keep automation files** in sync with root
5. **Document new tasks** in README

## 🛠️ Customization

### Adding New Tasks
1. **Edit**: `project_files/automation/Makefile`
2. **Edit**: `project_files/automation/tasks.ps1`
3. **Sync**: Run `.\sync_automation.ps1`
4. **Test**: Run the new task

### Adding New Pre-commit Hooks
1. **Edit**: `project_files/automation/.pre-commit-config.yaml`
2. **Sync**: Run `.\sync_automation.ps1`
3. **Install**: Run `uv run pre-commit install`

## 📖 Documentation

- **Make Documentation**: https://www.gnu.org/software/make/
- **Pre-commit Documentation**: https://pre-commit.com/
- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **Black Documentation**: https://black.readthedocs.io/ 