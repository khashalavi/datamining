# setup_env.ps1 - Creates the uv environment for SNLI download

Write-Host "=== Creating SNLI Environment ===" -ForegroundColor Green

# Create project directory
$projectDir = "snli-project"
if (!(Test-Path $projectDir)) {
    New-Item -ItemType Directory -Path $projectDir | Out-Null
}
Set-Location $projectDir

# Create pyproject.toml
@"
[project]
name = "snli-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "datasets>=3.0.0",
    "huggingface-hub>=0.20.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]
"@ | Set-Content -Path "pyproject.toml"

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
uv venv --python 3.13

# Activate and install
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1
uv pip install -e .

# Register the environment as Jupyter kernel
python -m ipykernel install --user --name snli-env --display-name "SNLI Environment"

Write-Host "`n=== Environment ready! ===" -ForegroundColor Green
Write-Host "Activate with:  .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "Jupyter kernel registered as: 'SNLI Environment'" -ForegroundColor Cyan