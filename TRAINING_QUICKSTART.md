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

### Terminalde ilerleme (RunPod / SSH)

Script iki faz yazdirir:

- **Faz 1 — Ozellik cache (CSV):** metadata’daki videolar icin satir ozeti (`toplam` / `cache’te hazir` / `islenecek kalan`), ardindan tqdm ile `Ozellik cikarma` cubugu (kalan süre ETA), faz sonunda yazilan/toplam süre özeti.
- **Faz 2 — Fusion:** tqdm ile epoch cubugu (`val_bce`, `best`).

Ozellikler `feature_cache*.csv` icine yazildikça dosya buyur; yarıda kesilirsen yeniden kosuda sadece **eksik** videolar işlenir (baslik yazilmis bos cache icin `--reset-cache` gerekebilir).

Cubuklar bazen uzak konsolda kiriliyorsa: `--no-progress` ile sadece satir satir log.

```bash
python train/train_fusion_from_metadata.py \
  --metadata-csv data/avlips_metadata.csv \
  --cache-csv data/feature_cache.csv \
  --out-model models/fusion_model.json
```

RunPod’da dataset repoda degilse once veriyi `/workspace/` altına kopyalayip `metadata_builder.py` icin `--dataset-root` yolunu buna gore ver.

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

---

## RunPod (GPU pod) — bastan egitim

Asagidaki yol, RunPod’da **PyTorch + CUDA** sablonu (or. `runpod/pytorch` veya resmi PyTorch imaji) ve veri setinin podda erisilebilir bir dizinde oldugu varsayimina gore yazildi.

### 0) Veri seti

AVLips kokunde `0_real/` ve `1_fake/` klasorleri olmali. Ornek yollar:

- Network volume: `/workspace/AVLips`
- Veya repoyla ayni disk: `/workspace/data/AVLips`

Veriyi pod’a **volume**, **RunPod dosya yukleme** veya `rsync`/`scp` ile koy; repoda veri yok.

**`FileNotFoundError: Expected ... 0_real`** ise ya veri henüz `/workspace/AVLips` altında degil ya da kok bir kademe ic ice (or. `/workspace/AVLips/AVLips`). `metadata_builder.py` bir alt klasorde `0_real`/`1_fake` arar; yine olmuyorsa podda kok bul:

```bash
ls -la /workspace
find /workspace -maxdepth 5 -type d \( -name 0_real -o -name 1_fake \) 2>/dev/null
```

Cikan yolun **ust** dizinini `export DATASET_ROOT=...` yap (icinde hem `0_real` hem `1_fake` olan klasor).

#### Internetten indirme (resmi AVLips v1.0, ~9 GB sikistirilmis)

Kaynak: [LipFD — release “AVLips dataset v1.0”](https://github.com/AaronComo/LipFD/releases/tag/dataset) (NeurIPS 2024 calismasi; veri **Google Drive** uzerinde).

RunPod’da ornek:

```bash
cd /workspace
source .venv/bin/activate 2>/dev/null || python3 -m venv .venv && source .venv/bin/activate
pip install -q gdown

# Google Drive dosya ID (release sayfasindaki baglanti ile ayni)
gdown --fuzzy "https://drive.google.com/file/d/1fEiUo22GBSnWD7nfEwDW86Eiza-pOEJm/view?usp=sharing"

# Inen dosya adi zip/7z/rar olabilir; ornek:
unzip -q *.zip -d AVLips_unpacked 2>/dev/null || (apt-get update && apt-get install -y unzip && unzip -q *.zip -d AVLips_unpacked)

find /workspace/AVLips_unpacked -maxdepth 5 -type d -name '0_real'
```

Cikan `.../0_real` yolunun **bir ust dizini** `DATASET_ROOT` olur (genelde `.../AVLips` veya `.../AVLips v1.0/AVLips`). Indirme basarisiz olursa tarayicidan ayni release linkini acip manuel export da denenebilir; buyuk dosyada `gdown` bazen tekrar calistirmayi gerektirir.

### 1) Ortam (tek blok, bash)

Podda `bash` ac; proje ve venv:

```bash
cd /workspace
git clone https://github.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi.git
cd Multimodal-Deepfake-Tespit-Sistemi

apt-get update && apt-get install -y ffmpeg   # ffmpeg yoksa (imaja gore sudo gerekebilir)

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

PyTorch: **CUDA surumunu** podunun CUDA’sina gore [pytorch.org](https://pytorch.org/get-started/locally/) uzerinden sec. Ornek (CUDA 12.4 uyumlu teker — surumu ihtiyaca gore degistir):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r deploy/requirements-runtime.txt
```

Gorsel checkpoint kullanacaksan `models/faceforensics/full/full_c23.p` kopyala veya:

```bash
export DF_VISUAL_MODEL_PATH=/workspace/full_c23.p
```

### 2) Metadata

`DATASET_ROOT` ve gercek kokunu kendi yoluna cevir:

```bash
export DATASET_ROOT=/workspace/AVLips   # 0_real ve 1_fake burada

python data_tools/metadata_builder.py \
  --dataset-root "$DATASET_ROOT" \
  --out-csv data/avlips_metadata.csv \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --seed 42
```

### 3) Egitim (terminalden izleme)

Uzun kosular icin `tmux new -s train` sonra:

```bash
source /workspace/Multimodal-Deepfake-Tespit-Sistemi/.venv/bin/activate
cd /workspace/Multimodal-Deepfake-Tespit-Sistemi

python train/train_fusion_from_metadata.py \
  --metadata-csv data/avlips_metadata.csv \
  --cache-csv data/feature_cache.csv \
  --out-model models/fusion_model.json \
  --lr 0.05 \
  --epochs 500
```

Hizli smoke test:

```bash
python train/train_fusion_from_metadata.py \
  --metadata-csv data/avlips_metadata.csv \
  --cache-csv data/feature_cache_dev.csv \
  --out-model models/fusion_model_dev.json \
  --max-per-split 40 \
  --epochs 120
```

Cikti: `data/feature_cache.csv` (artan cache), `models/fusion_model.json`. Uzak SSH’da cubuk kirilirsa `--no-progress` ekle.

### 4) Sonuclari disari alma

- `scp` veya RunPod dosya paneli ile `models/fusion_model.json` ve istege bagli `data/feature_cache.csv` indirilebilir.
- Volume kullaniyorsan cache bir sonraki podda da kullanilir; metadata ayni yollarla yazildiysa egitim kaldigi yerden eksik videolari tamamlar.

