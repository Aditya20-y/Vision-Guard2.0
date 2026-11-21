# VisionGuard Windows PowerShell Startup Script
Param(
    [switch]$InstallDeps,
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
Write-Host "VisionGuard — Industrial Visual Anomaly Detection Platform" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python 3.11 virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1

if ($InstallDeps) {
    Write-Host "Installing VisionGuard Python requirements..." -ForegroundColor Yellow
    pip install --upgrade pip
    pip install -r requirements.txt
}

if ($RunTests) {
    Write-Host "Running pytest verification suite..." -ForegroundColor Yellow
    pytest -q
}

Write-Host "Launching VisionGuard Streamlit Dashboard..." -ForegroundColor Green
streamlit run dashboard/app.py
