# Setup script for quantum reservoir computing environment
# Run from the quantum_reservoir folder:
#   cd "C:\Users\rr642bg\OneDrive - Technicka univerzita v Kosiciach\Desktop\quantum reservoir\quantum_reservoir"
#   .\setup_venv.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Fix git lock if present ---
$lockFile = ".git\index.lock"
if (Test-Path $lockFile) {
    Write-Host "Removing stale git lock file..." -ForegroundColor Yellow
    Remove-Item $lockFile -Force
}

# --- Remove empty/failed venv if present ---
if (Test-Path "venv") {
    Write-Host "Removing old venv directory..." -ForegroundColor Yellow
    Remove-Item "venv" -Recurse -Force
}

# --- Create virtual environment ---
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

Write-Host "Upgrading pip..." -ForegroundColor Cyan
.\venv\Scripts\python.exe -m pip install --upgrade pip --quiet

Write-Host "Installing packages from requirements.txt..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install -r requirements.txt

Write-Host "Registering Jupyter kernel..." -ForegroundColor Cyan
.\venv\Scripts\python.exe -m ipykernel install --user --name qrc_venv --display-name "Python (QRC)"

# --- Git: initial commit ---
Write-Host "Committing files to git..." -ForegroundColor Cyan
git config user.email "rusnak.renata@gmail.com"
git config user.name "Renata"
git add .gitignore requirements.txt setup_venv.ps1 datasets/
git commit -m "Add benchmark dataset notebooks, requirements.txt, .gitignore, and venv setup script"

Write-Host ""
Write-Host "All done!" -ForegroundColor Green
Write-Host "To start JupyterLab: .\venv\Scripts\jupyter lab"
Write-Host "Select kernel: Python (QRC)"
