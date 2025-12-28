
# 🎭 Multimodal Deepfake Detection System  
**Vision • Audio • Lip-Sync • Explainability • LLM Support**

Bu proje, **deepfake videolarını çoklu modalite (görüntü + ses + dudak–ses senkronu)** üzerinden analiz eden,  
**açıklanabilir (XAI)** ve **uçtan uca çalışan** bir yapay zeka sistemidir.

Amaç yalnızca *“fake mi?”* demek değil;  
**“neden fake / neden gerçek?”** sorusuna **kanıta dayalı açıklama** üretmektir.

---

## 🎯 Multimodal Deepfake Tespit Sistemi — Vitrin

> 📌 Aşağıda sunulan tüm görseller ve çıktılar, geliştirilen sistemin  
> **gerçek zamanlı çalışması sırasında elde edilen çıktılardır**.

---

## 🖥️ 1) Arayüz (UI)

Sistem, yüklenen video için **görüntü**, **ses** ve **dudak–ses senkronizasyonu** analizlerini  
**paralel (eş zamanlı)** olarak çalıştırır ve tüm çıktıları **tek bir panelde** sunar.

![Arayüz](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/1_arayüz.png)

*Multimodal analiz sonuçlarının, skorların ve açıklamaların tek bir arayüzde sunulması.*

---

## 🔍 2) Explainability — Grad-CAM

CNN tabanlı görsel modelin karar verirken odaklandığı yüz bölgeleri  
**Grad-CAM** yöntemi ile görselleştirilmiştir.  
Isı haritaları, modelin deepfake kararını verirken hangi bölgeleri **ayırt edici** bulduğunu gösterir.

| Deepfake Örneği | Gerçek (BN) |
|-----------------|-------------|
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/grandcam.png) | ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/grandcam_bn.jpg) |

*Sol: Deepfake videoda anormal odaklanmalar — Sağ: Gerçek videoda daha dengeli aktivasyonlar.*

Çıktı arayüz yorumu: Neden, hangi parametre yüzünden sorularına cevap 
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/gradcam_cıktı_arayüz_yorumu.png) |
---

## 👄 3) Ağız Kareleri (BN vs DF)

Gerçek (**BN**) ve deepfake (**DF**) videolardan çıkarılan ağız bölgesi kareleri gösterilmektedir.  
Dudak hareketleri ile ses arasındaki **zamansal uyumsuzluklar**, deepfake videolarda belirginleşir.

| Gerçek (BN) | Deepfake (DF) |
|-------------|---------------|
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/agız_kareleri_bn.png) | ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/agız_kareleri_df.png) |

*Gerçek videolarda doğal dudak hareketleri, deepfake videolarda ise senkron bozuklukları görülür.*

---

## 🧠 4) LLM Yorumları (Neden Deepfake?)

Modelden elde edilen sayısal skorlar, büyük dil modeli (**LLM**) tarafından yorumlanarak  
kullanıcıya **“neden deepfake / neden gerçek?”** sorusuna yönelik **metinsel gerekçeler** sunar.
| Gerçek (BN) | Deepfake (DF) |
|-------------|---------------|
| ![](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/df_gercek_video_yanıtı.png) | ![LLM Yanıtı](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/df_llm_yanıtı.jpeg)



*LLM tarafından üretilen insan-dostu açıklama.*

![LLM Yönlendirme](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/hangi_dosyada_llm_bagladım.png)

*Hangi analiz çıktılarının LLM’e yönlendirildiğini gösteren akış.*

---

## 📄 5) PDF Çıktısı (Otomatik Rapor)

Tüm analiz sonuçları, görseller ve açıklamalar otomatik olarak  
**PDF raporu** hâline getirilir ve dışa aktarılır.

![PDF Çıktısı](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/pdf_cıktısı.png)

*Otomatik oluşturulan, arşivlenebilir analiz raporu.*

---

## 📊 6) Parametre Grafikleri

Model skorları ve eşik değerleri grafiksel olarak sunularak  
karar mekanizmasının **şeffaflığı** artırılır.

![Parametre Grafikleri](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/parametre_grafikleri.png)

*Modalite bazlı skor dağılımları ve karar eşikleri.*

---

## 🎥 7) Demo (Uçtan Uca)

Gerçek zamanlı çalışan sistemin uçtan uca kullanımını gösteren demo video:

👉 **[Demo videosunu izlemek için tıklayın](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/assets/demo.mp4)**

---

## 🧠 Sistem Mimarisi

```text
Video Input
   ├── Visual Analysis (CNN + Grad-CAM)
   ├── Audio Analysis (ASR + Artefact Detection)
   ├── Lip-Sync Analysis (Mouth / Audio Alignment)
   └── LLM-based Explanation
            ↓
        Fusion Layer
            ↓
   Final Deepfake Probability + Explanation
```

---

## 🔍 Modaliteler

### 🎥 Görüntü (Visual)
- CNN / Xception tabanlı model  
- Frame-level analiz  
- Grad-CAM ile açıklanabilirlik  

### 🔊 Ses (Audio)
- ASR tabanlı çözümleme  
- GAN artefakt analizi  

### 👄 Dudak–Ses Senkronu
- Ağız bölgesi tespiti  
- Audio–visual hizalama  

### 🧠 LLM Yorumlama
- Skorların metinsel açıklaması  
- “Neden fake / neden gerçek?” cevabı  

---

## 📂 Proje Yapısı

```bash
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
│
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

## ⚙️ Kurulum

```bash
pip install -r requirements.txt
```

## ▶️ Çalıştırma

```bash
streamlit run src/app.py
```

---

## 📊 Çıktılar

- Final deepfake skoru (0–1)  
- Modalite bazlı skorlar  
- Görsel açıklamalar  
- Metinsel gerekçe  

---

## 👩‍💻 Geliştirici

**Büşra Mina Al**  
Artificial Intelligence & Industrial Engineering  
Ostim Teknik Üniversitesi

---

> Trustworthy AI requires explainable decisions.



# TEST

![TEST](https://raw.githubusercontent.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/main/screenshots/1_arayuz.png)

