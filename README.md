# Multimodal Deepfake Tespit Sistemi

**Görüntü · Ses · Biyomekanik · Dudak–ses senkronu · Açıklanabilirlik · LLM desteği**

Bu proje, videoları **çoklu modalite** üzerinden analiz eden ve mümkün olduğunca **kanıta dayalı açıklama** üreten bir araştırma sistemidir. Amaç yalnızca “fake mi?” demek değil; modal skorlar, ısı haritaları ve isteğe bağlı LLM metniyle **“neden şüpheli / neden gerçek?”** sorusuna yanıt vermektir.

**Depo:** [github.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi](https://github.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi)

---

## Öne çıkanlar

| | |
|---|---|
| **Görev** | Çok modlu deepfake / manipülasyon tespiti |
| **Modal skorlar** | `Sv` görüntü, `Sa` ses artefaktı, `Sl` dudak–ses, `Sb` göz kırpma, `Sh` baş pozu, `Sf` füzyon çıktısı |
| **XAI** | Grad-CAM (görsel odak), skor tabanlı yorumlar |
| **Arayüz** | Streamlit: paralel analiz, grafikler, PDF |
| **Ölçeklenebilir değerlendirme** | Özellik önbelleği, lojistik / HistGB füzyon, 5-fold CV, Platt kalibrasyonu |

---

## Vitrin (UI ve çıktılar)

> Görseller, sistemin çalışır halinden alınmıştır.

### Arayüz

Yüklenen video için görüntü, ses ve dudak–ses analizleri paralel çalışır; çıktılar tek panelde toplanır.

![Arayüz](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/1_arayüz.png)

### Explainability — Grad-CAM

| Deepfake | Gerçek (BN) |
|---|---|
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/grandcam.png) | ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/grandcam_bn.jpg) |

![Grad-CAM yorumu](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/gradcam_cıktı_arayüz_yorumu.png)

**Not:** Grad-CAM ve LLM çıktıları *post-hoc* açıklamalardır; nedensel garanti taşımazlar.

### Ağız ROI karşılaştırması

| Gerçek (BN) | Deepfake (DF) |
|---|---|
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/agız_kareleri_bn.png) | ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/agız_kareleri_df.png) |

### LLM yorumları ve PDF

| Gerçek (BN) | Deepfake (DF) |
|---|---|
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/df_gercek_video_yanıtı.png) | ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/df_llm_yanıtı.jpeg) |

![LLM bağlantısı](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/hangi_dosyada_llm_bagladım.png)

![PDF çıktısı](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/pdf_cıktısı.png)

![Parametre grafikleri](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/parametre_grafikleri.png)

### Demo videosu

[Demo (MP4)](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/assets/demo.mp4)

---

## Sistem mimarisi (güncel)

```
Video
 ├─ Görsel (CNN + Grad-CAM)          → Sv
 ├─ Ses (ASR + artefakt)             → Sa
 ├─ Dudak–ses hizası                 → Sl
 ├─ Göz kırpma / baş (biyomekanik) → Sb, Sh
 └─ Öğrenilmiş füzyon (eğitimle)    → Sf  (+ eşik / olasılık)
        ↓
 Streamlit / JSON / PDF + isteğe bağlı LLM metni
```

**Füzyon:** Arayüzde hâlâ ağırlıklı `src/fusion.py` yorumları kullanılabilir; **toplu değerlendirme ve CLI tahmini** için eğitilmiş model (`models/fusion_model.json`, lojistik) kullanılır. Araştırma hattında **HistGradientBoosting + Platt kalibrasyonu** ile daha güçlü metrikler elde edilir (aşağıda özet, ayrıntı `results/v2/SUMMARY.md`).

---

## AVLips ölçekli değerlendirme (v2 özeti)

AVLips v1.0 düzenine uygun **`data/avlips_metadata.csv`** ile yapılan çalışmada (7602 video; sabit train/val/test bölünmesi) modal skorlar `data/feature_cache*` ile önbelleğe alınır. Raporlanan metrikler `results/v2/` altındaki JSON dosyaları ve [results/v2/SUMMARY.md](results/v2/SUMMARY.md) ile uyumludur.

| Model | Test dengeli doğruluk | Test AUC |
|-------|------------------------|----------|
| Lojistik füzyon (üretim/CLI varsayılanı, Sl+Sa) | ~0.53 | ~0.51 |
| HistGB (6 özellik: Sv, Sl, Sb, Sh, Sa, Sf) | ~0.56 | ~0.61 |
| 5-fold CV — HistGB (ortalama ± std) | **0.594 ± 0.011** | **0.623 ± 0.011** |

Platt kalibrasyonu ile test **ECE** yaklaşık **0.025** seviyesine iner. Üretim CLI şu an lojistik `fusion_model.json` bekler; HistGB+Platt’ı tahmin boru hattına taşımak için `eval/train_fusion_histgb.py` ve `results/v2/fusion_calibration.json` yol haritası `SUMMARY.md` Bölüm 7’de özetlenmiştir.

**Teknik derinlik:** [docs/MAKALE_TEKNIK_MULTIMODAL_DEEPFAKE.md](docs/MAKALE_TEKNIK_MULTIMODAL_DEEPFAKE.md)

---

## Proje yapısı

```
Multimodal-Deepfake-Tespit-Sistemi/
├── src/                    # Streamlit app, analiz, modal skorlar, fusion yardımcıları
├── infer/                  # predict_video.py, batch_predict_json.py
├── train/                  # train_fusion_from_metadata.py, auto_select_fusion_model.py, tune_fusion_hparams.py
├── eval/                   # evaluate_fusion, fusion_cv, fusion_calibration, train_fusion_histgb, hata raporları, figürler
├── data_tools/             # metadata_builder, refresh_sl_cache, Sl diff araçları
├── scripts/                # PowerShell boru hatları (fusion, Sl yenileme)
├── data/                   # avlips_metadata.csv (repoda); feature_cache* .gitignore
├── docs/                   # teknik makale (Türkçe)
├── results/                # v2 metrikler, figürler, hata analizi çıktıları
├── models/                 # fusion_model.json tipik olarak .gitignore; _auto_search deney JSON’ları repoda olabilir
├── network/models/         # TransferModel vb.
├── sample_data/, assets/, screenshots/
├── requirements.txt
├── run_demo.py, server.py
└── README.md
```

---

## Modalite özeti

| Sinyal | Yaklaşık rol | Çıktı |
|--------|----------------|--------|
| Görüntü | FaceForensics++ tabanlı omurga, Grad-CAM | `Sv`, ısı haritası |
| Ses | Whisper ASR + artefakt skoru | `Sa` |
| Dudak–ses | ROI + senkron tutarlılığı (v2’de varyans kapısı, MA, hız karışımı) | `Sl` |
| Biyomekanik | Göz kırpma, baş pozu | `Sb`, `Sh` |
| Füzyon | Eğitilmiş sınıflandırıcı | `Sf`, olasılık |
| LLM | İstemle metin | İnsan-okur gerekçe |

---

## Kurulum

- Python **3.10+** (Windows / Linux; GPU isteğe bağlı)
- [FFmpeg](https://ffmpeg.org/) PATH’te olmalı (video/ses işleme)

```bash
pip install -r requirements.txt
```

`requirements.txt` içinde **Streamlit** ve **Plotly** (arayüz grafikleri) yer alır. PDF raporu için isteğe bağlı: `pip install reportlab`. Görsel omurga ağırlıkları repoda yoktur; `.gitignore` içinde `models/faceforensics*` ve `models/*.zip` vb. tanımlıdır.

---

## Çalıştırma

**Web arayüzü**

```bash
streamlit run src/app.py
```

**Tek video — eğitilmiş füzyon JSON ile tahmin**

```bash
python infer/predict_video.py --video path/to/video.mp4 --model-json models/fusion_model.json
```

`models/fusion_model.json` yerelde üretilmelidir (`train/auto_select_fusion_model.py` veya ilgili script’ler). Özellik önbelleği `data/feature_cache.csv` büyük olabilir ve **Git’e alınmaz**.

---

## Araştırma / yeniden üretim (kısa yol haritası)

1. `data/avlips_metadata.csv` ve videolar hazır.
2. Özellik çıkarma ve CSV önbellek (proje script’leri / `data_tools`).
3. `train/train_fusion_from_metadata.py` ve `train/auto_select_fusion_model.py` ile lojistik füzyon ve arama raporu.
4. `eval/train_fusion_histgb.py`, `eval/fusion_cv.py`, `eval/fusion_calibration.py` ile HistGB, CV ve kalibrasyon.

Ayrıntılı komutlar ve tablolar: [results/v2/SUMMARY.md](results/v2/SUMMARY.md).

---

## Repoda olmayan / gizlenen dosyalar

`.gitignore` özeti: `data/feature_cache*`, `models/*.json` (kök), `models/*.zip`, FaceForensics ağırlık klasörleri, `logs/`, `*.log`, `venv` vb. Bu nedenle klon sonrası **ağırlık ve fusion_model** ayrıca yerleştirilmelidir.

---

## Veri ve etik

Yeni bir kamusal veri seti yayınlanmamıştır; akademik çalışmada **AVLips** (ve benzeri) kaynaklı bölünmeler ve metadata kullanılır. FaceForensics++ ağırlıkları yalnızca çıkarım içindir ve lisanslarına tabidir.

---

## Uygulama alanları

Dijital adli tıp, medya doğrulama, hukuki ön inceleme, çok modlu güvenilir AI araştırmaları.

---

## Gelecek çalışmalar

- `Sv` için hedef veri setinde ince ayar; çapraz veri seti (DFDC, Celeb-DF) testleri
- SyncNet benzeri öğrenilmiş lip-sync gömlekleri
- HistGB+Platt’ın `infer/predict_video.py` ile aynı API’de sunulması

---

## Katkılar (özet)

Paralel multimodal analiz, Grad-CAM, dudak–ses hizası, isteğe bağlı LLM katmanı, PDF raporu; v2 ile birlikte ölçeklenebilir füzyon değerlendirmesi, Sl iyileştirmesi ve kalibre olasılık raporları.

---

## Geliştirici

**Büşra Mina Al**  
Artificial Intelligence & Industrial Engineering, Ostim Teknik Üniversitesi

---

## Lisans

Bu proje **akademik ve araştırma amaçlı** kullanım içindir.

> Trustworthy AI requires explainable decisions.
