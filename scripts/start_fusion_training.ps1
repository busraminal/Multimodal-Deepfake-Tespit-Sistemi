# Tam AVLips fusion egitimi: Faz 1 ozellik cache + Faz 2 logistic fusion.
# Yeni PowerShell penceresinde calistirin; tqdm ilerlemesi + log dosyasi.

# WinPS 5.x: dogrudan & python ... 2>&1 bazen NativeCommandError kaydi dusurur (stderr).
# cmd /c ile stderr stdout'a birlesir; PowerShell yanlislikla hata sanmaz.

# --- Kosum modu ---
# "auto"  : Tum ozellik kombinasyonlarini dene; val_balanced_acc + --pos-weight-auto ile en iyiyi models\fusion_model.json'a yaz (onerilen).
# "single": Tek bir logistic fusion (cache dolu degilse ilk kosuda ozellikler cikarilir). Ozellik listesini gerekiyorsa guncelleyin.
$FusionMode = "auto"

# single modunda kullanilacak ozellikler (models\fusion_model.json icindeki feature_names ile uyumlu tutun)
$SingleFeatures = "Sv,Sl,Sb,Sh"

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
Write-Host "FusionMode: $FusionMode" -ForegroundColor Cyan

if ($FusionMode -eq "auto") {
    $cmdLine = @"
cd /d `"$Root`" && `"$py`" -u train\auto_select_fusion_model.py --metadata-csv data\avlips_metadata.csv --cache-csv data\feature_cache.csv --out-model models\fusion_model.json --report-json models\fusion_model_search_report.json --pos-weight-auto --epochs 1200 --lr 0.03 --l2 0.01 2>&1
"@
}
else {
    $cmdLine = @"
cd /d `"$Root`" && `"$py`" -u train\train_fusion_from_metadata.py --metadata-csv data\avlips_metadata.csv --cache-csv data\feature_cache.csv --out-model models\fusion_model.json --features $SingleFeatures --standardize --pos-weight-auto --epochs 1200 --lr 0.03 --l2 0.01 2>&1
"@
}

& cmd.exe /c $cmdLine | Tee-Object -FilePath $Log

Write-Host "`nBitis." -ForegroundColor Green
Write-Host "  Model: models\fusion_model.json"
if ($FusionMode -eq "auto") {
    Write-Host "  Rapor: models\fusion_model_search_report.json"
}
Read-Host "Kapatmak icin Enter"
