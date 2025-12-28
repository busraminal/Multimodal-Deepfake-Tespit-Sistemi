
# 🎭 Multimodal Deepfake Detection System  
**Vision • Audio • Lip-Sync • Explainability • LLM Support**

Bu proje, **deepfake videolarını çoklu modalite (görüntü + ses + dudak-ses senkronu)** üzerinden analiz eden, **açıklanabilir (XAI)** ve **uçtan uca çalışan** bir yapay zeka sistemidir.

Amaç yalnızca *“fake mi?”* demek değil,  
**“neden fake / neden gerçek?”** sorusuna **kanıta dayalı açıklama** üretmektir.

---
> 📌 Bu bölümde sunulan tüm görseller, geliştirilen multimodal deepfake tespit sisteminin **gerçek zamanlı çalışması sırasında elde edilen çıktılardır**.

---

### 🖥️ Arayüz
Sistem arayüzü; yüklenen video için görsel, işitsel ve senkronizasyon analizlerini paralel olarak çalıştırır ve sonuçları tek bir panelde sunar.

![UI](screenshots/1_arayuz.png)

---

### 👄 Ağız Kareleri (BN vs DF)
Gerçek (BN) ve deepfake (DF) videolardan çıkarılan ağız bölgesi kareleri.  
Bu karşılaştırma, dudak hareketlerindeki tutarsızlıkların görsel olarak incelenmesini sağlar.

<img src="screenshots/agiz_kareleri_bn.png" width="280"/>
<img src="screenshots/agiz_kareleri_df.png" width="280"/>

---

### 🔍 Explainability (Grad-CAM)
CNN tabanlı görsel modelin karar verirken odaklandığı yüz bölgeleri Grad-CAM ile görselleştirilmiştir.  
Isı haritaları, modelin şüpheli bölgeleri nasıl tespit ettiğini açıklamaya yardımcı olur.

<img src="screenshots/grandcam.png" width="280"/>
<img src="screenshots/grandcam_bn.jpg" width="280"/>

---

### 🧠 LLM Yorumları
Model çıktıları, büyük dil modeli (LLM) tarafından yorumlanarak **“neden deepfake?”** sorusuna insan-dostu açıklamalar üretir.  
Ayrıca hangi analiz çıktısının LLM’e yönlendirildiği de gösterilmektedir.

![LLM](screenshots/df_llm_yaniti.jpeg)
![Routing](screenshots/hangi_dosyada_llm_bagladim.png)

---

### 📊 Parametre Grafikleri
Analiz sürecinde elde edilen skorlar ve eşik değerleri grafiksel olarak sunularak model davranışı daha şeffaf hale getirilmiştir.

![Params](screenshots/parametre_grafikleri.png)

---

### 🎥 Demo
Gerçek zamanlı çalışan sistemin uçtan uca kullanımını gösteren örnek demo videosu.  
[Demo videosunu izlemek için tıklayın](screenshots/demo_videosu.mp4)


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




https://github.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/raw/main/assets/demo.mp4

