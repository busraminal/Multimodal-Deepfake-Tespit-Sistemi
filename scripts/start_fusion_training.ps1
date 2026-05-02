# Tam AVLips fusion egitimi: Faz 1 ozellik cache + Faz 2 logistic fusion.
# Yeni PowerShell penceresinde calistirin; tqdm ilerlemesi + log dosyasi.

# WinPS 5.x: dogrudan & python ... 2>&1 bazen NativeCommandError kaydi dusurur (stderr).
# cmd /c ile stderr stdout'a birlesir; PowerShell yanlislikla hata sanmaz.
$ErrorActionPreference = "Continue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
# FFmpeg her clip icin cok log basar; tqdm (N/7602, ETA) satirini kaybetmemek icin:
$env:FFMPEG_QUIET = "1"
$Log = Join-Path $Root ("training_" + (Get-Date -Format "yyyyMMdd_HHmm") + ".log")

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Once: python -m venv .venv ve pip install ..."
    exit 1
}

Write-Host "Log: $Log" -ForegroundColor Cyan

$cmdLine = @"
cd /d `"$Root`" && `"$py`" -u train\train_fusion_from_metadata.py --metadata-csv data\avlips_metadata.csv --cache-csv data\feature_cache.csv --out-model models\fusion_model.json --lr 0.05 --epochs 500 2>&1
"@
& cmd.exe /c $cmdLine | Tee-Object -FilePath $Log

Write-Host "`nBitis. Models: models\fusion_model.json" -ForegroundColor Green
Read-Host "Kapatmak icin Enter"
