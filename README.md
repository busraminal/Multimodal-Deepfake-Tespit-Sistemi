
# 🎭 Multimodal Deepfake Detection System  
**Vision • Audio • Lip-Sync • Explainability • LLM Support**

Bu proje, **deepfake videolarını çoklu modalite (görüntü + ses + dudak-ses senkronu)** üzerinden analiz eden, **açıklanabilir (XAI)** ve **uçtan uca çalışan** bir yapay zeka sistemidir.

Amaç yalnızca *“fake mi?”* demek değil,  
**“neden fake / neden gerçek?”** sorusuna **kanıta dayalı açıklama** üretmektir.

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



## 🎥 Demo Video

![Demo](assets/demo.gif)

▶️ Full video:  
https://github.com/busraminal/Multimodal-Deepfake-Tespit-Sistemi/raw/main/assets/demo.mp4

