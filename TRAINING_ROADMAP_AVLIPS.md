# Multimodal Deepfake Egitim Yol Haritasi (AVLips Odakli)

Bu dokuman, projeyi "yalnizca inference" seviyesinden cikarip egitilebilir, olculebilir ve uretime yaklasan bir hatta tasimak icin adim adim plandir.

## 1) Hedef ve Kapsam

- Hedef cikti: `REAL`, `FAKE`, `UNCERTAIN`
- Girdi: video (`.mp4`) + ses (`.wav`, varsa)
- Alt moduller:
  - `S_visual`: goruntu artefakt skoru
  - `S_sync`: lip-sync uyumsuzluk skoru
  - `S_mouth`: agiz acikligi/zamansal dogallik skoru
  - `S_audio`: ses artefakt skoru
  - `S_emotion`: duygu-zaman tutarlilik skoru
- Final karar: `Sf = sum(w_i * S_i)` + kalibrasyon + uncertain bandi

## 2) Mevcut AVLips Durumu (Bu Workspace Tarama Sonucu)

- `0_real`: **3396** video
- `1_fake`: **4206** video
- `wav`: **7602** ses dosyasi
- Toplam video: **7602**
- Veri boyutu (yaklasik):
  - Real: `2.94 GB`
  - Fake: `3.14 GB`
- Klasor yapisi: tek seviye (`0_real`, `1_fake`, `wav`), alt kimlik/speaker klasoru yok.

## 3) AVLips Yeterli mi?

Kisa cevap: **MVP icin yeterli, yuksek genelleme icin tek basina yetersiz**.

Neden:
- Arti:
  - Veri adedi guzel (7k+ video)
  - Real/Fake dagilimi cok bozuk degil
  - Lip-sync agiz temelli calismalar icin uygun bir baslangic
- Eksi:
  - Kimlik bazli ayrim bilgisi yok gorunuyor (leakage riski)
  - Tek dataset kaynakli overfit/domain bias riski
  - Compression, kamera, dil, senaryo cesitliligi sinirli olabilir
  - Gercek dunyada farkli deepfake araclarina genelleme garantisi dusuk

Sonuc:
- **Asama-1:** AVLips ile baseline egit.
- **Asama-2:** En az 1-2 harici dataset ile external test yap (DFDC, FaceForensics++, CelebDF vb.)

## 4) Teknik Mimari (Egitim)

### 4.1 Moduller

1. `visual_model_train.py`
   - Backbone: EfficientNet-B0 / Xception
   - Input: secili frame/clip
   - Cikti: `S_visual`

2. `sync_model_train.py`
   - Input: mouth motion embedding + audio embedding
   - Yontem: contrastive veya binary classifier
   - Cikti: `S_sync`

3. `mouth_dynamics_train.py`
   - Input: landmark tabanli mouth aspect ratio zaman serisi
   - Model: 1D-CNN veya BiLSTM
   - Cikti: `S_mouth`

4. `audio_model_train.py`
   - Input: log-mel spectrogram
   - Model: hafif CNN
   - Cikti: `S_audio`

5. `emotion_consistency.py`
   - Video ve ses emotion skorlari arasinda temporal tutarlilik
   - Cikti: `S_emotion`

6. `fusion_train.py`
   - Baslangic: sabit agirlik
   - Sonra: Logistic Regression / LightGBM ile ogrenilen fuzyon
   - Cikti: `Sf` + threshold

## 5) Veri Hazirlama Plani

1. Index olustur (`metadata.csv`)
   - Alanlar: `video_path,label,audio_path,duration,fps,resolution,split`
2. Quality filtreleri
   - cok dusuk cozum / bozuk dosya / ses yok durumlari etiketlenmeli
3. Split stratejisi
   - Zorunlu: en az `train/val/test`
   - Mümkünse: speaker/identity based split
   - Mümkün degilse: leakage riskini raporda acik belirt
4. Dengeleme
   - Class weighted loss veya sampler

## 6) Egitim Stratejisi

### Asama A - Baseline (1-2 hafta)
- Visual + Audio + Basit Fusion
- Hedef: calisan ve olculen pipeline

### Asama B - Lip + Mouth (1 hafta)
- `S_sync` ve `S_mouth` ekle
- Ablation: modullerin tek tek katkisini olc

### Asama C - Emotion + Kalibrasyon (1 hafta)
- `S_emotion` ekle
- Temperature scaling / Platt scaling
- `UNCERTAIN` bandi optimize et

## 7) Basari Olcutleri (Zorunlu)

- Accuracy tek basina yeterli degil
- Raporlanacak metrikler:
  - ROC-AUC
  - F1 (macro)
  - EER
  - Precision / Recall
  - Confusion Matrix
- Ek:
  - Dataset ici test
  - Cross-dataset test
  - Compression-robustness test

## 8) Uretim Hazirligi Kontrolleri

- Her tahmin icin:
  - `Sf`, alt skorlar, katkilar (feature importance)
  - en supheli kareler
  - "neden" ozeti
- OOD bayragi:
  - dusuk kalite/ses yoksa "guven dusuk" uyarisi
- Versiyonlama:
  - model versiyonu
  - threshold
  - agirliklar
  - egitim verisi hash/versiyon

## 9) Hemen Uygulanacak Gorev Listesi

1. `data/metadata_builder.py` yaz
2. `train/visual_train.py` baseline egitim scripti
3. `train/audio_train.py` baseline egitim scripti
4. `train/fusion_train.py` (LR tabanli)
5. `eval/evaluate.py` (AUC/F1/EER)
6. `infer/predict_video.py` (tek video, alt skor + final)
7. `reports/experiment_template.md` ile deney kaydi standardi

## 10) Pratik Karar

- Evet, AVLips ile hemen baslanabilir.
- Ama "yuksek dogruluk" hedefi icin tek basina AVLips'e guvenme.
- Dogru strateji:
  - once AVLips ile guclu baseline,
  - sonra harici test ve domain genisletme,
  - en son threshold/kalibrasyon ile stabil karar.

