# Quantune PowerShell Launcher
Set-Location $PSScriptRoot

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "                 Starting Quantune                " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check virtual environment
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "[INFO] Virtual environment not found. Setting it up..." -ForegroundColor Yellow
    python -m venv .venv
    & (Join-Path $PSScriptRoot ".venv\Scripts\pip.exe") install -r requirements.txt
}

# 2. Free port 8000 if occupied to prevent WinError 10013
$portProcess = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($portProcess) {
    Write-Host "[INFO] Freeing occupied port 8000 (PID: $($portProcess.OwningProcess))..." -ForegroundColor Yellow
    Stop-Process -Id $portProcess.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# 3. Open browser after 2 seconds
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:8000"
} | Out-Null

# 4. Start Quantune
Write-Host "[INFO] Quantune starting at http://localhost:8000" -ForegroundColor Green
Write-Host "[INFO] Press Ctrl+C to stop the server." -ForegroundColor Green
Write-Host ""

& $pythonExe -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
