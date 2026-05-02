# Tam AVLips fusion egitimi: Faz 1 ozellik cache + Faz 2 logistic fusion.
# Yeni PowerShell penceresinde calistirin; tqdm ilerlemesi + log dosyasi.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
$Log = Join-Path $Root ("training_" + (Get-Date -Format "yyyyMMdd_HHmm") + ".log")

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Once: python -m venv .venv ve pip install ..."
    exit 1
}

Write-Host "Log: $Log" -ForegroundColor Cyan

& $py -u .\train\train_fusion_from_metadata.py `
    --metadata-csv data\avlips_metadata.csv `
    --cache-csv data\feature_cache.csv `
    --out-model models\fusion_model.json `
    --lr 0.05 `
    --epochs 500 `
    2>&1 | Tee-Object -FilePath $Log

Write-Host "`nBitis. Models: models\fusion_model.json" -ForegroundColor Green
Read-Host "Kapatmak icin Enter"
