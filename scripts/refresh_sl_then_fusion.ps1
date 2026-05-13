# Sl sutununu yenile (konuşma kapisi ile), ardından fusion özellik araması.
# Uzun surebil; yalnızca tek bir Sl yenileme islemi ayni anda calistirin (dosya carpismasi olmasin).

$ErrorActionPreference = "Continue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
$env:FFMPEG_QUIET = "1"

# Bitince lr/l2 ince ayari (models\fusion_hparam_tune_report.json)
$TuneAfterAutoSelect = $false

$Log = Join-Path $Root ("sl_refresh_fusion_" + (Get-Date -Format "yyyyMMdd_HHmm") + ".log")
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Once: python -m venv .venv ve pip install ..."
    exit 1
}

Write-Host "Log: $Log" -ForegroundColor Cyan

$step1 = @"
cd /d `"$Root`" && `"$py`" -u data_tools\refresh_sl_cache.py --backup 2>&1
"@
Write-Host "1/2 refresh_sl_cache --backup ..." -ForegroundColor Yellow
& cmd.exe /c $step1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) {
    Write-Host "Sl yenileme hata kodu: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

$step2 = @"
cd /d `"$Root`" && `"$py`" -u train\auto_select_fusion_model.py --metadata-csv data\avlips_metadata.csv --cache-csv data\feature_cache.csv --out-model models\fusion_model.json --report-json models\fusion_model_search_report.json --pos-weight-auto --epochs 1200 --lr 0.03 --l2 0.01 2>&1
"@
Write-Host "2/2 auto_select_fusion_model ..." -ForegroundColor Yellow
& cmd.exe /c $step2 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) {
    Write-Host "Auto-select hata kodu: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

if ($TuneAfterAutoSelect) {
    $step3 = @"
cd /d `"$Root`" && `"$py`" -u train\tune_fusion_hparams.py --pos-weight-auto --epochs 2000 2>&1
"@
    Write-Host "3/3 tune_fusion_hparams ..." -ForegroundColor Yellow
    & cmd.exe /c $step3 | Tee-Object -FilePath $Log -Append
}

Write-Host "`nTamam: models\fusion_model.json guncellendi (ve istenirse tune)." -ForegroundColor Green
Write-Host "Yedek cache: data\feature_cache.csv.bak (refresh --backup ile)" -ForegroundColor Gray
Read-Host "Kapatmak icin Enter"
