# Yerel kurulum: venv + bagimliliklar + (istege bagli) metadata
# Calistir:  powershell -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Pip = Join-Path $Root ".venv\Scripts\pip.exe"

if (-not (Test-Path $Py)) {
    Write-Host "venv olusturuluyor..."
    python -m venv .venv
}

& $Py -m pip install --upgrade pip
Write-Host "PyTorch (CPU)..."
& $Pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
Write-Host "Runtime paketleri..."
& $Pip install -r (Join-Path $Root "deploy\requirements-runtime.txt")
& $Pip install "grad-cam>=1.4.0"

$Meta = Join-Path $Root "data\avlips_metadata.csv"
if (-not (Test-Path $Meta)) {
    Write-Host "metadata yok, ornek yol ile olusturuluyor (kendi yolunu duzenle)..."
    $Avlips = "C:\Users\busra\Desktop\projeler\df\df_video\AVLips v1.0\AVLips"
    & $Py (Join-Path $Root "data_tools\metadata_builder.py") --dataset-root $Avlips --out-csv $Meta
}

Write-Host "Tamam. Arayuz: .\.venv\Scripts\python.exe -m streamlit run src\app.py"
Write-Host "Egitim (ornek): .\.venv\Scripts\python.exe train\train_fusion_from_metadata.py --metadata-csv data\avlips_metadata.csv --cache-csv data\feature_cache.csv --out-model models\fusion_model.json --max-per-split 50"
