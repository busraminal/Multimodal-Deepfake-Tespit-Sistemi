# Training Quickstart (AVLips)

Bu dosya, eklenen egitim hattini hizlica calistirman icin minimal komutlari verir.

Windows icin proje kokunde: `.\.venv\Scripts\python.exe` kullan (torch/mediapipe vb. icin).

## Local gelistirme (Windows, Docker yok)

Tum komutlari **`Multimodal-Deepfake-Tespit-Sistemi`** klasorunun icinde acik PowerShell / CMD ile calistir.

### 1) Sanal ortam

```powershell
cd C:\Users\busra\Desktop\projeler\df\df_video\Multimodal-Deepfake-Tespit-Sistemi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) PyTorch + kutuphaneler

**CPU** (en basit):

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r deploy\requirements-runtime.txt
```

**NVIDIA GPU** varsa: [PyTorch kurulum sayfasindan](https://pytorch.org/get-started/locally/) CUDA’li komutu secip once `torch`/`torchaudio` kur, sonra ayni `requirements-runtime` satiri.

### 3) FFmpeg

Ses/video cikarma icin sistemde **ffmpeg** olmali. [ffmpeg builds](https://www.gyan.dev/ffmpeg/builds/) ile indirip `C:\ffmpeg\bin` altina koy; kod bu yolu otomatik kullanir (`PATH`’e eklemen de yeterli).

### 4) Arayuz (Streamlit)

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\app.py
```

Tarayici: `http://localhost:8501`

### 5) Tek komut kurulum (alternatif)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1
```

### 6) Gorsel model (opsiyonel ama onemli)

Checkpoint yoksa goruntu skoru heuristik calisir. Agirlik dosyani `models\faceforensics\full\full_c23.p` konumuna koy veya ortam degiskeni:

`DF_VISUAL_MODEL_PATH=C:\tam\yol\full_c23.p`

Bundan sonra asagidaki **Metadata / Egitim / Tahmin** adimlariyla local egitim hattina devam edebilirsin.

**Not:** Tum AVLips (~7602 video) icin `feature_cache` uretmek cok uzun surer; hizli deneme icin `train_fusion_from_metadata.py` ile `--max-per-split N` kullan. Tam egitim icin `--max-per-split` verme.

---

## 1) Metadata olustur

```bash
python data_tools/metadata_builder.py ^
  --dataset-root "C:\Users\busra\Desktop\projeler\df\df_video\AVLips v1.0\AVLips" ^
  --out-csv data/avlips_metadata.csv ^
  --train-ratio 0.70 ^
  --val-ratio 0.15 ^
  --seed 42
```

## 2) Fusion modelini egit

Not: Bu adim her video icin `src/analyze_video.py` calistirarak ozellik cikarir; ilk kosu uzun surebilir.

```bash
python train/train_fusion_from_metadata.py ^
  --metadata-csv data/avlips_metadata.csv ^
  --cache-csv data/feature_cache.csv ^
  --out-model models/fusion_model.json ^
  --lr 0.05 ^
  --epochs 500
```

Onceki yarim kalmis cache'i silip bastan uretmek icin `--reset-cache` ekle.

Hizli test icin:

```bash
python train/train_fusion_from_metadata.py ^
  --metadata-csv data/avlips_metadata.csv ^
  --cache-csv data/feature_cache_dev.csv ^
  --out-model models/fusion_model_dev.json ^
  --max-per-split 30 ^
  --epochs 100
```

`--max-per-split` sadece train/val/test icin secilen satirlarda ozellik cikarir (tum 7602 videoyu taramaz).

Windows PowerShell eski surumlerde `&&` yerine komutlari ayri satirlarda veya `;` ile calistir.

## 3) Modeli degerlendir

```bash
python eval/evaluate_fusion.py ^
  --metadata-csv data/avlips_metadata.csv ^
  --cache-csv data/feature_cache.csv ^
  --model-json models/fusion_model.json
```

## 4) Tek videoda tahmin

```bash
python infer/predict_video.py ^
  --video "C:\Users\busra\Desktop\projeler\df\df_video\AVLips v1.0\AVLips\0_real\0.mp4" ^
  --model-json models/fusion_model.json ^
  --out-json results/predict_0_real_0.json
```

## Cikti Dosyalari

- `data/avlips_metadata.csv`: split bilgisi
- `data/feature_cache.csv`: video bazli modal skorlar (`Sv,Sl,Sb,Sh,Sa`)
- `models/fusion_model.json`: egitilen fusion agirliklari + esik + metrikler
- `results/*.json`: tek video tahmin ciktilari

## Notlar

- Bu surum, mevcut kod tabanindaki heuristik/pretrained sinyalleri birlestirir.
- Yani baseline hizli kurulur, sonra moduller (visual/audio/lipsync) ayri ayri guclendirilebilir.
- Daha iyi genelleme icin sonraki asamada harici dataset ile cross-dataset test eklenmelidir.

## Docker (Hetzner / Linux)

Sadece kendi bilgisayarinda calisacaksan bu bolumu atlayabilirsin.

Detay: [deploy/README.md](deploy/README.md)

```bash
docker compose up -d --build
```

Sunucuda `models/` icine `full_c23.p` (veya `DF_VISUAL_MODEL_PATH`) koy; bos volume ile ezilirse gorsel model yuklenmez.

