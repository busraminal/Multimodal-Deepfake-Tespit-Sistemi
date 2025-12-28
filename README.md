# 🎭 Multimodal Deepfake Detection System
**Vision • Audio • Lip-Sync • Explainability • LLM Support**

Bu proje, **deepfake videolarını çoklu modalite (görüntü + ses + dudak–ses senkronu)** üzerinden analiz eden,
**açıklanabilir (XAI)** ve **uçtan uca çalışan** bir yapay zeka sistemidir.

Amaç yalnızca *“fake mi?”* demek değil;
**“neden fake / neden gerçek?”** sorusuna **kanıta dayalı açıklama** üretmektir.

---

## 🔎 Quick Facts
- **Görev:** Multimodal Deepfake Tespiti
- **Modaliteler:** Görüntü, Ses, Dudak–Ses Senkronu
- **Açıklanabilirlik:** Grad-CAM + LLM
- **Çıktılar:** Skor + İnsan-okur açıklama + PDF raporu
- **Durum:** Araştırma / Prototip

---

## 🎯 Vitrin
> Aşağıdaki tüm görseller ve çıktılar, sistemin **gerçek zamanlı** çalışması sırasında elde edilmiştir.

### 🖥️ Arayüz (UI)
Sistem, yüklenen video için **görüntü**, **ses** ve **dudak–ses senkronizasyonu** analizlerini
**paralel** olarak çalıştırır ve tüm çıktıları **tek bir panelde** sunar.

![Arayüz](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/1_arayüz.png)

---

### 🔍 Explainability — Grad-CAM
CNN tabanlı görsel modelin karar verirken odaklandığı yüz bölgeleri **Grad-CAM** ile görselleştirilir.

| Deepfake | Gerçek (BN) |
|---|---|
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/grandcam.png) | ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/grandcam_bn.jpg) |

![Arayüz Yorumu](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/gradcam_cıktı_arayüz_yorumu.png)

**Açıklanabilirlik Kapsamı**
- Açıklamalar **post-hoc**tur; nedensel değildir.
- Grad-CAM ayırt edici bölgeleri vurgular, mutlak doğruluk göstermez.
- LLM açıklamaları model çıktılarıyla koşulludur.

---

### 👄 Ağız Kareleri (BN vs DF)
Gerçek ve deepfake videolardan çıkarılan ağız ROI kareleri karşılaştırılır.

| Gerçek (BN) | Deepfake (DF) |
|---|---|
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/agız_kareleri_bn.png) | ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/agız_kareleri_df.png) |

---

### 🧠 LLM Yorumları
Sayısal skorlar, LLM tarafından **“neden fake / neden gerçek?”** sorusuna yanıt verecek şekilde metne dönüştürülür.

| Gerçek (BN) | Deepfake (DF) |
|---|---|
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/df_gercek_video_yanıtı.png) | ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/df_llm_yanıtı.jpeg) |

![LLM Akış](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/hangi_dosyada_llm_bagladım.png)

---

### 📄 PDF Rapor
Analiz sonuçları otomatik olarak **PDF raporu**na dönüştürülür.

![PDF](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/pdf_cıktısı.png)

---

### 📊 Parametre Grafikleri
Skorlar ve eşikler grafiksel olarak sunulur.

![Grafikler](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/parametre_grafikleri.png)

---

### 🎥 Demo
👉 **[Demo videosu](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/assets/demo.mp4)**

---

## 🧠 Sistem Mimarisi
```
Video
 ├─ Görsel Analiz (CNN + Grad-CAM) → Sv
 ├─ Ses Analizi (ASR + Artefakt) → Sa
 ├─ Dudak–Ses Senkronu (AV Alignment) → Sl
 └─ Karar Füzyonu → Sf
        ↓
 Açıklanabilir Skor + Metinsel Gerekçe
```

**Füzyon:** `Sf = α·Sv + (1−α)·Sl`,  `α ∈ [0.3, 0.7]`

---

## 🔍 Modaliteler
- **Görüntü:** Xception/CNN, frame-level analiz, Grad-CAM
- **Ses:** ASR tabanlı çözümleme, artefakt analizi
- **Dudak–Ses:** Ağız ROI, AV hizalama
- **LLM:** Skorların metinsel açıklaması

---

## 📂 Proje Yapısı
```
deepfake_project/
├── src/
│   ├── app.py
│   ├── media_io.py
│   ├── visual_model.py
│   ├── visual_score.py
│   ├── gradcam_utils.py
│   ├── audio_artefact.py
│   ├── asr_text.py
│   ├── lip_sync.py
│   ├── mouth_detect.py
│   ├── mouth_embed.py
│   ├── fusion.py
│   ├── llm_client.py
│   └── biomech.py
├── network/models/
├── assets/demo.mp4
├── sample_data/
├── server.py
├── run_demo.py
├── rag_knowledge.json
├── requirements.txt
└── README.md
```

---

## 📂 Veri Seti
Bu projede yeni bir veri seti oluşturulmamıştır.
Sistem, **demo ve nitel analiz** amacıyla sınırlı sayıda gerçek ve deepfake video ile çalışır.

- Görsel model: **FaceForensics++ pretrained** ağırlıkları
- Modeller yalnızca **inference** amaçlıdır ve repoda paylaşılmaz

---

## 🧩 Modality–Responsibility Mapping
| Modality | Yöntem | Çıktı |
|---|---|---|
| Görüntü | CNN + Grad-CAM | Frame skorları + ısı haritaları |
| Ses | ASR + Artefakt | Ses özgünlük skoru |
| Dudak–Ses | AV Alignment | Senkron tutarlılık skoru |
| LLM | Prompted reasoning | Metinsel açıklama |

---

## ♻️ Reproducibility
- Python **>= 3.10**
- Windows / Linux test edildi
- GPU opsiyonel (CPU destekli)

---

## ⚙️ Kurulum
```bash
pip install -r requirements.txt
```

## ▶️ Çalıştırma
```bash
streamlit run src/app.py
```

---

## 🎯 Uygulamalar
- Dijital adli bilişim
- Medya doğrulama
- Hukuki ön inceleme
- Akademik multimodal AI araştırmaları

## 🚀 Gelecek Çalışmalar
- Göz kırpma & baş-poz anomali tespiti
- Zamansal transformer füzyonu
- DFDC / FaceForensics++ benchmarkları

---

## 📊 Çıktılar
- Final deepfake skoru (0–1)
- Modalite bazlı skorlar
- Görsel açıklamalar
- Metinsel gerekçe

---

## ✨ Katkılar
- Paralel multimodal analiz hattı
- Grad-CAM ile frame-level açıklanabilirlik
- Dudak–ses hizalama ile senkron uyumsuzluk tespiti
- LLM tabanlı semantik açıklama katmanı
- Otomatik PDF raporlama

---

## 👩‍💻 Geliştirici
**Büşra Mina Al**  
Artificial Intelligence & Industrial Engineering  
Ostim Teknik Üniversitesi

---

## 📝 Lisans
Bu proje **akademik ve araştırma amaçlı** kullanım içindir.

> Trustworthy AI requires explainable decisions.
