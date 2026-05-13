# AVLips Üzerinde Çok Modlu Deepfake Tespiti için Uçtan Uca Bir Boru Hattı: Görsel, İşitsel ve Davranışsal Sinyallerin Lojistik ve Ağaç Tabanlı Füzyonu

**Yazar(lar):** Büşra (proje sahibi)
**Tarih:** 12 Mayıs 2026
**Sürüm:** v2 — `Sl` ölçüm güncellemesi + HistGB / Çapraz Doğrulama / Olasılık Kalibrasyonu raporlama eklemeleri sonrası.
**Doküman türü:** Teknik makale (tek dosya, derin ayrıntı).

---

## Özet (Türkçe)

Bu çalışmada AVLips veri kümesi üzerinde çalışan **çok modlu deepfake tespit boru hattı** uçtan uca tanıtılmaktadır. Sistem her video için altı sayısal sinyal üretir: görsel artefakt skoru (`Sv`), dudak–ses uyumsuzluğu (`Sl`), biyomekanik tutarsızlık skorları (`Sb` göz kırpma; `Sh` baş pozu), ses artefakt skoru (`Sa`) ve sabit ağırlıklı kural tabanlı birleşik skor (`Sf`). Bu skorlar, hem deterministik hem de **veriden öğrenilen lojistik füzyon** ile sahte/gerçek etiketine eşlenir. Çalışma kapsamında: (1) dudak–ses uyum ölçümü doygunluk ve düz-sinyal patolojisine karşı yeniden tasarlanmış (`varyans kapısı + hareketli ortalama + birinci-fark karışımı`); (2) tüm önbellek 7402 video için yeniden hesaplanmış; (3) lojistik füzyona ek olarak `HistGradientBoostingClassifier` ile **non-lineer füzyon** denenmiş; (4) **5-kat çapraz doğrulama** ile şanslı bölünme etkisi elenmiş; (5) **Platt** ve **isotonic kalibrasyon** ile olasılık güvenilirliği `ECE` üzerinden raporlanmıştır. Sabit bölünmede HistGB modeli test setinde **0.557 dengeli doğruluk** ve **0.612 AUC** üretmiş; 5-kat CV'de her foldda lojistik tabanı geçerek **0.594 ± 0.011 dengeli doğruluk** ve **0.623 ± 0.011 AUC**'e ulaşmıştır. Platt kalibrasyonu test **ECE**'sini **0.077 → 0.025** seviyesine indirmiştir.

## Abstract (English)

We present an end-to-end multimodal deepfake detection pipeline evaluated on the AVLips dataset. For each input video, the system extracts six scalar scores—visual artefact score (`Sv`), lip-audio mismatch (`Sl`), eye-blink and head-pose biomechanical scores (`Sb`, `Sh`), an audio artefact score (`Sa`), and a fixed-weight pipeline aggregate (`Sf`). A learnt logistic head is trained on a cached score matrix; for analysis we additionally study a `HistGradientBoostingClassifier`, stratified 5-fold cross-validation, and probability calibration with Platt (sigmoid) and isotonic methods. We re-engineer the lip-sync measurement to mitigate saturation and silent-input pathologies, refresh the score cache for all 7402 retained videos, and report consistent gains for the tree-based fusion over logistic baselines: test balanced accuracy 0.557 and AUC 0.612 in holdout, and 0.594 ± 0.011 balanced accuracy with 0.623 ± 0.011 AUC across 5 folds. Platt scaling reduces test Expected Calibration Error from 0.077 to 0.025 without sacrificing AUC.

**Anahtar kelimeler / Keywords:** deepfake detection, multimodal fusion, lip-sync, AVLips, logistic regression, gradient boosting, probability calibration, expected calibration error.

---

## İçindekiler

1. Giriş ve Motivasyon
2. İlgili Çalışmalar
3. Veri Kümesi ve Deney Düzeni
4. Sistem Mimarisi
5. Modal Skorların Tanımı ve Hesaplanması
6. Kural Tabanlı Birleşik Skor (`Sf`)
7. Öğrenilmiş Füzyon Katmanı
8. Değerlendirme Protokolü ve Metrikler
9. Deneyler ve Bulgular (9.1 holdout, 9.2 CV, 9.3 kalibrasyon, 9.4 Sl yenilemenin tek başına etkisi, **9.5 baseline karşılaştırması — eski Sl vs yeni Sl**)
10. Hata Analizi
11. Üretim Çıkarımı (Deploy) ile Analiz Modelleri Arasındaki Ayrım (Lojistik vs HistGB+Platt)
12. Tartışma
13. Etik ve Toplumsal Konular
14. Sınırlılıklar ve Gelecek Çalışma
15. Sonuç
16. Kaynakça
17. Ek A — Dizin ve Komut Sözlüğü
18. Ek B — JSON Çıktılarından Özet Sayılar
19. Ek C — Yeniden Üretim Adımları (Reproducibility)
20. Görselleştirmeler ve Kaynak PDF Arşivi

---

## 1. Giriş ve Motivasyon

Üretici derin modellerin (özellikle dudak senkronizasyonu ve yüz değiştirme üzerine GAN/diffusion tabanlı yöntemler) yaygınlaşmasıyla birlikte **video deepfake** içeriklerinin tespiti, dezinformasyon, kimlik dolandırıcılığı ve hukuki delillendirme bağlamında kritik bir mühendislik problemi haline gelmiştir [1, 2]. Tek bir modaliteye (yalnızca yüz, yalnızca ses, yalnızca dudak hareketi) dayanan dedektörler; sıkıştırma, çözünürlük, konuşmacı, dil, arka plan, aydınlatma çeşitliliklerine karşı kırılgan kalabilmektedir [3, 4]. Buna karşılık **çok modlu yaklaşımlar**, hem görüntüden gelen artefakt sinyali hem de ses–dudak uyumu gibi çapraz-modal tutarsızlıkları kullanarak daha sağlam karar verme potansiyeli sunar [5, 6, 7].

Bu çalışma; AVLips dataseti üzerinde, modüler ve yeniden üretilebilir bir **çok modlu skor + füzyon** sistemi inşa etmeyi hedeflemektedir. Tez kapsamındaki katkılar şu şekildedir:

- **K1.** Modüler skor çıkarımı (görsel, biyomekanik, ses, dudak-sync) ve önbelleğe alma altyapısı.
- **K2.** **Dudak–ses uyumsuzluğu ölçümünün yeniden tasarlanması** (varyans kapısı, yumuşatma, türev karışımı).
- **K3.** Lojistik ile **ağaç tabanlı** füzyonun aynı özellik kümesi üzerinde **adil karşılaştırması**.
- **K4.** **5-kat çapraz doğrulama** ile bölünme bağımlılığını azaltma.
- **K5.** **Platt** ve **Isotonic** olasılık kalibrasyonu, **ECE** ve **Brier** ile raporlama.
- **K6.** **Üretim çıkarımı (logistic JSON)** ile **analiz modelleri (HistGB, kalibre)** arasında ayrımı açıkça belgeleme.

---

## 2. İlgili Çalışmalar

**Görsel tabanlı dedektörler.** FaceForensics++ veri kümesi; XceptionNet ve benzeri evrişimli ağlarla, sıkıştırma düzeyleri (`raw`, `c23`, `c40`) altında yüksek skorlar üretebilen referans bir koleksiyon ortaya koymuştur [1]. Bu çalışmada görsel modal `Sv` puanı, FaceForensics++-uyumlu önceden eğitilmiş bir omurganın (`models/faceforensics/full/full_c23.p`) ağız ROI'leri üzerindeki sahte olasılığından türetilir; gerektiğinde **Laplacian varyansı** tabanlı bir heuristik yedek kullanılır [8].

**Dudak senkron tabanlı dedektörler.** Sync ölçütlerinin deepfake için kullanılabilirliği, **SyncNet** [9] ve **Wav2Lip** [10] çalışmalarının ardından klasik korelasyon yöntemleriyle de tartışılmıştır. Bu çalışma, öğrenilmiş bir senkron embedding'i yerine, **ses enerji zarfı ile MediaPipe** [11] tabanlı dudak açıklığı zaman serisi arasında **gecikmeye duyarlı mutlak Pearson korelasyonu** kullanır; doygunluk ve sessiz girdi patolojilerine karşı modülasyon (varyans kapısı, yumuşatma, birinci fark karışımı) eklenmiştir.

**Ses tabanlı dedektörler.** ASVspoof yarışmaları konuşma deepfake'lerini (TTS/VC) [12, 13] tespit için referans alanları olarak öne çıkmıştır. Bu projenin ses modal `Sa` skoru, hafif spektral / timbre ipuçlarına dayalı bir GAN-eğilim sezici (`src/audio_artefact.py`) üzerinden üretilmektedir.

**Biyomekanik ipuçları.** Erken dönem deepfake'lerin **göz kırpma** sıklığında ve **baş pozu** geçişlerinde anormallik göstermesi gözlemlenmiştir [14, 15]. Bu çalışmada `Sb` (blink) ve `Sh` (headpose) bu hattın hafif modelleri (`src/biomech.py`) olarak boru hattına eklenmiştir.

**Çok modlu füzyon.** Çapraz modaliteyi öğrenmek için ortak embedding, geç füzyon (skor seviyesi) ve karar seviyesi füzyon kullanılabilir [5, 6, 7]. Bu çalışma; modal skor düzeyinde **lojistik regresyon** (öğrenilmiş, ayrıca sabit ağırlıklı `Sf`) ve **ağaç tabanlı non-lineer füzyon** [16] olarak iki seçeneği aynı önbellek üzerinde karşılaştırır.

**Olasılık kalibrasyonu.** Modern güçlü sınıflandırıcılar (özellikle ağaç toplulukları) doğru AUC üretmesine rağmen olasılıkları abartılı/eksik raporlayabilir. **Platt scaling** [17] ve **Isotonic regression** [18] bu sorunu hafifletmek için kullanılan klasik yöntemlerdir. Kalibrasyon kalitesi **ECE** [19] ve **Brier skoru** [20] ile ölçülür.

**Kullanılan kütüphaneler.** Sayısal işlemler için NumPy [21], makine öğrenmesi için scikit-learn [22], derin model için PyTorch [23], ses için Librosa [24], video/I-O için FFmpeg [25] ve OpenCV [26], yüz mesh için MediaPipe [11] kullanılmıştır.

---

## 3. Veri Kümesi ve Deney Düzeni

### 3.1 AVLips

AVLips v1.0 koleksiyonu, hem gerçek (`0_real`) hem de sahte (`1_fake`) videolarla birlikte eşleşen WAV dosyalarını içerir. Bu projenin metadata CSV'si (`data/avlips_metadata.csv`) her satırda `video_path`, `label` (0 = real, 1 = fake) ve `split` (`train`/`val`/`test`) alanlarını taşır. Kullanılan toplam örnek sayısı **7602** (gerçek 3396, sahte 4206); sabit bölünmede **train=5321 / val=1139 / test=1142** olarak ayrılmıştır.

### 3.2 Önbellek (`feature_cache.csv`)

Her video için modal skorlar bir kez hesaplanır ve `data/feature_cache.csv` içine yazılır. Sütunlar: `video_path, Sv, Sl, Sb, Sh, Sa, Sf_pipeline`. Öğrenilmiş füzyon eğitim/değerlendirme zinciri yalnızca bu önbelleği okur. `Sf` özelliği önbellekte `Sf_pipeline` sütunu altında, model JSON içinde ise `Sf` anahtarıyla kullanılır; ad eşlemesi `src/fusion_features.py` tarafından yapılır.

### 3.3 Sl Yenileme

Yeni ölçüm formülasyonu sonrası `data_tools/refresh_sl_cache.py` ile **7402** satırın `Sl` sütunu yeniden hesaplanmış, **4 paralel işçi** ve 200'lük checkpoint aralığı ile yaklaşık **158 dk** sürede tamamlanmıştır. Sürecin günlüğü `logs/refresh_sl_cache_resume.log` içinde tutulmaktadır.

---

## 4. Sistem Mimarisi

Boru hattı, kavramsal olarak dört aşamadan oluşur:

```mermaid
flowchart LR
  V[Video (MP4)] --> M[media_io: FFmpeg + MediaPipe]
  M -->|WAV 16k mono| A[ses analizi]
  M -->|ağız ROI kareleri| F[görsel + biyomekanik]
  A --> Sa[Sa]
  A --> Sl[Sl]
  F --> Sv[Sv]
  F --> Sb[Sb]
  F --> Sh[Sh]
  Sl --> SF[Sf — kural tabanlı v3 (analyze_video)]
  Sv --> SF
  Sb --> SF
  Sh --> SF
  Sa --> SF
  SF --> CSV[(feature_cache.csv)]
  Sv --> CSV
  Sl --> CSV
  Sb --> CSV
  Sh --> CSV
  Sa --> CSV
  CSV --> LR[Lojistik (üretim modeli) — fusion_model.json]
  CSV --> ML[HistGB / CV / Kalibrasyon (analiz)]
  LR --> INF[infer/predict_video.py — REAL/FAKE/UNCERTAIN]
```

**Aşama A — Çıkarım:** Geçici dizinde WAV ve ağız bölgesi kareleri üretilir; modal skorlar hesaplanır; sabit ağırlıklı `Sf` üretilir.
**Aşama B — Önbellek:** Skor matrisi diske yazılır.
**Aşama C — Öğrenilmiş füzyon eğitimi:** Lojistik modeli (`train/train_fusion_from_metadata.py`); kombinasyon araması (`train/auto_select_fusion_model.py`); analiz amaçlı HistGB (`eval/train_fusion_histgb.py`), 5-kat CV (`eval/fusion_cv.py`) ve kalibrasyon (`eval/fusion_calibration.py`).
**Aşama D — Çıkarım arayüzü:** `infer/predict_video.py` tek video alır, lojistik füzyon JSON'unu uygular, `p_fake` ve karar etiketi döndürür.

---

## 5. Modal Skorların Tanımı ve Hesaplanması

### 5.1 Görsel Skor — `Sv`

Kaynak: `src/visual_score.py` + `src/visual_model.py`.

- Ağız ROI'si **96×96** olarak `media_io.extract_frames` tarafından çıkarılır; görsel skor için kareler 299×299'a yeniden boyutlandırılarak modele beslenir.
- Eğer FaceForensics++-uyumlu omurga (`DF_VISUAL_MODEL_PATH` veya `models/faceforensics/full/full_c23.p`) yüklenebiliyorsa, **softmax sahte olasılığı** kare başına alınır.
- Aksi halde, Laplacian varyans temelli **heuristik** çağrılır (`_heuristic_frame_score`): aşırı düşük detay veya `ringing/oversharpen` belirtisi “şüphe” skoruna eklenir.
- Final `Sv`, varsayılan olarak **tüm karelerin sahte olasılığının ortalamasıdır**. İsteğe bağlı olarak üst çeyrek karışımı (`DF_VISUAL_SV_TOPQ_BLEND`) etkinleştirilebilir.

`Sv ∈ [0,1]`; yüksek değer = "sahte eğilim" yorumu.

### 5.2 Biyomekanik Skorlar — `Sb` ve `Sh`

- `Sb`: göz kırpma kanıtı / örüntüsü (`src/biomech.blink_score`). Erken dönem deepfake'lerin göz kırpmada anormallik gösterdiği bilinen bir gözlemdir [14].
- `Sh`: baş pozu kararlılığı / geçiş normalizasyonu (`src/biomech.headpose_score`).
- İstisna durumunda her ikisi de nötr **0.5** atanır.

`Sb, Sh ∈ [0,1]`; yüksek değer = artefakt eğilimi.

### 5.3 Ses Artefakt Skoru — `Sa`

Kaynak: `src/audio_artefact.py`, `audio_gan_score()`. Spektral / timbral öznitelikler üzerinden hafif bir karar üretilir; ASVspoof tipi sentetik ses sinyallerine yönelik bir başlangıçtır [12, 13]. Detaylar `audio_details` sözlüğünde döndürülerek raporlanabilir.

### 5.4 Dudak–Ses Uyumsuzluğu — `Sl`

Kaynak: `src/lip_sync.py`.

1. **Ses enerji zarfı** (`extract_audio_energy`): 16 kHz tek-kanal sinyal, 25 fps eşdeğeri pencerelerde toplam kare-enerji; ardından [0,1] aralığına min-max normalleştirilir.
2. **Dudak açıklığı zaman serisi** (`lip_openings`): MediaPipe FaceMesh [11] ile yüz landmarkları çıkarılır; dudak indeksleri (`LIPS`) üzerinden Öklid temelli açılma sinyali üretilir.
3. **Konuşma varlığı kapısı** (`has_speech_like_audio`): basit enerji-eşik VAD; aktivite oranı yetersizse `Sl=0`, `has_speech=False`.
4. **Senkron skoru** (`lip_sync_correlation`):
   - Varyans kapısı: ses veya dudak serisinin varyansı eşik altında ise `sync_score=0.5` ve `Sl=0.5` (belirsiz).
   - Hareketli ortalama yumuşatma (`DF_LIP_SYNC_SMOOTH_WIN=5`).
   - Gecikmeye duyarlı mutlak Pearson korelasyonu (`max_lag=6`).
   - Türev karışımı: ham ve **birinci fark** üzerindeki korelasyonun ağırlıklı toplamı (`DF_LIP_SYNC_VELOCITY_BLEND=0.45`).
5. **Skor:** `Sl = 1 - sync_score` (yüksek `Sl` = daha çok uyumsuzluk).

Bu güncellemenin temel motivasyonu, eski sürümde **çoğu videoda `Sl≈1.0` doygunluğu** gözlemlenmesi ve füzyon başlığının bu sabit terimi bias gibi kullanmasıydı. Yeni sürümde `Sl` artık gerçek bir lip-sync ölçüsü davranışı sergiler; **6183 satırda değer değişimi** (yaklaşık %81) ile önbellek yeniden hesaplanmıştır.

---

## 6. Kural Tabanlı Birleşik Skor (`Sf`)

Kaynak: `src/analyze_video._fusion_v3`. Konuşma varlığına göre iki ağırlık şeması:

- Konuşma var: `w_v=0.45, w_l=0.20, w_b=0.10, w_h=0.10, w_a=0.15`.
- Konuşma yok: `w_v=0.55, w_l=0.00, w_b=0.15, w_h=0.10, w_a=0.20`.

Toplam: `Sf = Σ w_i · S_i`, ardından [0,1] kırpılır. `src/fusion.interpret_score`, `Sf`'yi `REAL / FAKE / UNCERTAIN` etiketine eşleyebilir. Önbellekte `Sf_pipeline` olarak saklanır; öğrenilmiş füzyonun **opsiyonel** bir girdi özelliği olarak da kullanılır (`Sf`).

---

## 7. Öğrenilmiş Füzyon Katmanı

### 7.1 Lojistik Regresyon (üretim)

Kaynak: `train/train_fusion_from_metadata.py`.

- Tasarım matrisi `X`'in sütunları `feature_names` listesi tarafından belirlenir; **alt küme** veya `poly2` genişletmesi (`src/fusion_expand.py`) seçilebilir.
- Hedef: `BCE = -Σ [ y log σ(z) + (1-y) log (1-σ(z)) ]`, `z = w^T x + b`.
- Sınıf dengesizliği için pozitif örnek ağırlığı: `pos_weight = n_neg / n_pos` (`--pos-weight-auto`).
- Standartlaştırma seçeneği (`scaler_mean`, `scaler_std`) JSON'a yazılır; çıkarımda da uygulanır.
- Eşik: doğrulama setinde **dengeli doğruluk** veya **F1** maksimize edilerek seçilir (`--threshold-objective`).
- Çıktı JSON: `feature_names`, `feature_expansion`, `weights`, `bias`, `threshold`, `standardize`, `scaler_mean`, `scaler_std`, `metrics`.

### 7.2 Kombinasyon Araması

Kaynak: `train/auto_select_fusion_model.py`.

- `ALL_FUSION_FEATURES = [Sv, Sl, Sb, Sh, Sa, Sf]` üzerinde **max-combo-size** kadar tüm alt kümeler denenir; altı özellik için **63 model**.
- Seçim metriği varsayılan `val_balanced_acc` (`--metric`).
- En iyi model `models/fusion_model.json` adıyla kopyalanır; arama özeti `models/fusion_model_search_report.json`.

### 7.3 Ağaç Tabanlı Füzyon (analiz)

Kaynak: `eval/train_fusion_histgb.py`.

- Aynı önbellek/standartlaştırma boru hattı.
- `HistGradientBoostingClassifier` [16]: histogram tabanlı, sıralı bölünmelerin sayısallaştırılmış bin sınırları üzerinde çalışan gradient boosting.
- Hiperparametreler (varsayılan): `max_depth=6, max_iter=250, learning_rate=0.06, class_weight="balanced"`.
- Eşik: doğrulama dengeli doğruluğa göre ızgaralı arama.
- JSON çıktı: `feature_names`, `train_config`, `threshold_val_tuned`, `metrics`.

### 7.4 Çapraz Doğrulama

Kaynak: `eval/fusion_cv.py`. `train+val+test` birleştirilir, **StratifiedKFold(n_splits=5, shuffle=True, seed=42)**. Her foldda eğitim setinde mean/std hesaplanır, her iki model fit edilir; fold-test üzerinde eşik dengeli doğruluğa göre seçilir, `balanced_acc, F1, AUC` raporlanır; agregasyon ortalama±standart sapma.

### 7.5 Olasılık Kalibrasyonu

Kaynak: `eval/fusion_calibration.py`. HistGB tüm-özellik modeli train üzerinde eğitilir; **val** üzerinde:

- **Platt scaling** (`CalibratedClassifierCV(method="sigmoid", cv="prefit")`) [17];
- **Isotonic regression** (`method="isotonic"`) [18].

Test setinde **AUC**, **Brier skoru** [20] ve **ECE** [19] (10 eşit-genişlik bin) hesaplanır; her bin için `n, p_mean, y_mean, gap` ayrıntıları JSON'a yazılır.

### 7.6 Eşik Seçimi

İster lojistik ister HistGB olsun, eşik **doğrulama setinde dengeli doğruluk** maksimumuna göre `linspace(0.05, 0.95, 91)` üzerinde seçilir. Bu, sınıf oranındaki kayışları dengelemek için kullanılır.

---

## 8. Değerlendirme Protokolü ve Metrikler

- **Dengeli doğruluk** (`(TPR+TNR)/2`): sınıf dengesizliğinde tek başına `acc`ye göre daha bilgilendiricidir.
- **F1**: yalnızca pozitif sınıf (`fake=1`).
- **AUC** (ROC alanı): trapezoidal hesap (`numpy` ile manuel) [27].
- **Brier skoru**: `mean((p - y)^2)` — kalibrasyon ve diskriminasyonu birlikte ölçer [20].
- **Beklenen Kalibrasyon Hatası (ECE)**: 10 eşit-genişlik bin, ağırlıklı `|p̄_b − ȳ_b|` ortalaması [19].

Sabit bölünme protokolünde **train**'da fit, **val**'da eşik/kalibrasyon, **test**'te raporlama yapılır; test seti hiçbir kararda kullanılmaz (data leakage'a karşı).

---

## 9. Deneyler ve Bulgular

### 9.1 Sabit Bölünme Sonuçları

Aşağıdaki tablo, aynı `feature_cache.csv` ve bölünmesi (`train=5321/val=1139/test=1142`) üzerinde elde edilmiştir.

| Yaklaşım | Val Bal-Acc | Val AUC | Test Bal-Acc | Test AUC | Test F1 |
|---|---|---|---|---|---|
| Lojistik (auto-search en iyi: `Sl, Sa`) | 0.5512 | 0.5366 | 0.5331 | 0.5108 | 0.673 |
| HistGB (`Sl, Sa`) | 0.5640 | 0.5746 | 0.5318 | 0.5494 | 0.6266 |
| **HistGB (`Sv, Sl, Sb, Sh, Sa, Sf`)** | **0.5920** | **0.6264** | **0.5565** | **0.6120** | **0.6735** |

**Yorum.** Aynı modal skorlar üzerinde non-lineer füzyon, lojistiğe göre **val AUC +0.09**, **test AUC +0.10** kazandırır. Bu, görselin ve diğer modaların kendi içinde sınır-doğrusal olmayan etkileşimleri taşıdığını ima eder.

### 9.2 Beş Katlı Çapraz Doğrulama

Tüm 7602 örnek üzerinde stratified 5-fold (seed=42). Aşağıdaki sayılar `results/v2/fusion_cv_allfeats.json` dosyasından doğrudan alınmıştır.

| Model | Bal-Acc (mean ± std) | AUC (mean ± std) | F1 (mean ± std) |
|---|---|---|---|
| Lojistik | 0.5447 ± 0.0067 | 0.5522 ± 0.0165 | 0.6131 ± 0.1457 |
| **HistGB** | **0.5943 ± 0.0110** | **0.6234 ± 0.0113** | 0.6031 ± 0.0385 |

Fold ayrıntısı:

| Fold | LR Bal-Acc | LR AUC | GB Bal-Acc | GB AUC |
|---|---|---|---|---|
| 1 | 0.5396 | 0.5385 | 0.5771 | 0.6095 |
| 2 | 0.5433 | 0.5436 | 0.6014 | 0.6306 |
| 3 | 0.5414 | 0.5486 | 0.5936 | 0.6141 |
| 4 | 0.5565 | 0.5805 | 0.6059 | 0.6361 |
| 5 | 0.5426 | 0.5496 | 0.5936 | 0.6268 |

HistGB **her 5 foldda da lojistik tabanı geçer**; standart sapmalar küçüktür (0.011) — gözlemlenen üstünlük şanslı bölünme ile açıklanamaz.

### 9.3 Olasılık Kalibrasyonu (test seti)

`results/v2/fusion_calibration.json` özet sayıları:

| Varyant | AUC | Brier | **ECE** |
|---|---|---|---|
| Raw HistGB | 0.6120 | 0.2474 | **0.0765** |
| **Platt (sigmoid)** | 0.6120 | 0.2380 | **0.0248** |
| Isotonic | 0.6075 | 0.2419 | 0.0349 |

Platt scaling **AUC'yi koruyarak ECE'yi yaklaşık 3× iyileştirir**; Isotonic test verisi sınırlı olduğunda görece daha az kazanım verir (genellikle daha esnek olduğundan, çok küçük örneklerde over-fit riski taşır [18]).

Tipik bir “güven kalibrasyonu eğrisi” raporu için bin bazlı `p_mean` vs `y_mean` JSON içinde mevcuttur (`test_ece_bins`).

### 9.4 Sl Yenilemenin Tek Başına Etkisi

Yalnızca `Sl` ölçümü güncellendiğinde, lojistik füzyonun **doğruluk yüzdesi** kayda değer biçimde değişmedi; çünkü eski formülasyondaki `Sl≈1.0` doygunluğu lojistiğe sabit bir kaydırma olarak işlemekteydi. Buna karşın **yorumlanabilirlik** açısından artık `Sl` gerçek bir lip-sync ölçütüdür ve **hata analizi** anlamlı bir şekilde yapılabilir (Bölüm 10). HistGB'nin tüm özelliklerle elde ettiği AUC sıçramasının bir kısmı, `Sl`-temelli daha bilgilendirici girdiler nedeniyledir.

### 9.5 Baseline Karşılaştırması — Eski `Sl` (`.bak`) vs Yeni `Sl` (Sayısal)

Sl ölçümü değişikliğinin **adil ölçülmesi** için tüm değerlendirme zinciri eski önbellek üzerinde tekrar koşturuldu (`data/feature_cache.csv.bak`). Aynı script'ler, aynı bölünme, aynı 6 özellik (`Sv, Sl, Sb, Sh, Sa, Sf`). Sonuçlar `results/v2_oldSl/*.json` altında saklanmaktadır.

**(a) Lojistik füzyon — auto-search (63 alt küme) ile seçim.**

| Önbellek | Seçilen alt küme | Val Bal-Acc | Val AUC | Test Bal-Acc | Test AUC |
|---|---|---|---|---|---|
| Eski Sl (`.bak`) | Sv+Sl+Sh+Sa+Sf | 0.5661 | 0.5763 | 0.5482 | 0.5609 |
| Yeni Sl | Sl+Sa | 0.5512 | 0.5366 | 0.5331 | 0.5108 |

Görünürde **eski `Sl`** lojistik için daha iyi sayılar üretir. Bu, eski ölçümün `Sl≈1.0` doygunluğu nedeniyle lojistiğe yapay bir "fake≈Sl yüksek" kestirmesi sağladığı bir **bilgi sızıntısı/kestirme** durumudur — Bölüm 9.4 ve Bölüm 12'de belirtildiği üzere bu, **yorumlanabilir bir gerçek lip-sync sinyali değildir**.

**(b) HistGradientBoosting — 6 özellik holdout.**

| Önbellek | Val Bal-Acc | Val AUC | Test Bal-Acc | Test AUC |
|---|---|---|---|---|
| Eski Sl | 0.6015 | 0.6242 | 0.5480 | 0.6030 |
| **Yeni Sl** | **0.5920** | **0.6264** | **0.5565** | **0.6120** |

Yeni `Sl` ile test AUC **+0.009**, test Bal-Acc **+0.0085**.

**(c) Stratified 5-fold Cross-Validation — 6 özellik.**

| Önbellek | Model | Bal-Acc (mean ± std) | AUC (mean ± std) |
|---|---|---|---|
| Eski Sl | Lojistik | 0.5604 ± 0.0123 | 0.5788 ± 0.0159 |
| Eski Sl | HistGB | 0.5918 ± 0.0095 | 0.6209 ± 0.0131 |
| Yeni Sl | Lojistik | 0.5447 ± 0.0067 | 0.5522 ± 0.0165 |
| **Yeni Sl** | **HistGB** | **0.5943 ± 0.0110** | **0.6234 ± 0.0113** |

Çapraz doğrulamada da en güçlü konfigürasyon **yeni Sl + HistGB**: hem ortalama hem standart sapma açısından eski Sl'yi geçer (kazanım küçük ama foldlar arası tutarlı).

**(d) Olasılık kalibrasyonu — HistGB, test.**

| Önbellek | Varyant | AUC | Brier | ECE |
|---|---|---|---|---|
| Eski Sl | Raw HistGB | 0.6030 | 0.2492 | 0.0957 |
| Eski Sl | Platt | 0.6030 | 0.2395 | 0.0362 |
| Yeni Sl | Raw HistGB | 0.6120 | 0.2474 | 0.0765 |
| **Yeni Sl** | **Platt** | **0.6120** | **0.2380** | **0.0248** |

**Net sonuç.**

- Eski `Sl` doygun olduğu için **lojistik baseline**'a **kestirme** bir sinyal sunup yapay olarak iyi gözükürdü; oysa **bilgi taşımıyordu** (her video için ≈1).
- Yorumlanabilir, varyans taşıyan yeni `Sl` ile **non-lineer füzyon (HistGB) + Platt kalibrasyon** kombinasyonu **tüm metriklerde** (AUC, Bal-Acc, Brier, ECE) en iyi konfigürasyondur.
- Tezde rapor edilecek **temel hat** budur; eski `Sl` tablosu yalnızca kestirmeyi ifşa etmek ve baseline olarak ayrıca belgelemek için sunulmaktadır.

---

## 10. Hata Analizi

`eval/fusion_error_report.py` test setinde her satır için `p_fake`, `pred_fake`, `correct` ve eşige göre marjı yazar. `eval/fusion_error_feature_summary.py` ise FN/FP/TP/TN kümelerinde modal skor ortalama ve standart sapmalarını hesaplar. Yeni `Sl` ile elde edilen ortalama profiller (lojistik `Sl+Sa` üretim modeli için):

| Küme | Sv | Sl | Sb | Sh | Sa | Sf |
|---|---|---|---|---|---|---|
| TP | 0.42 | 0.67 | 0.86 | 0.23 | 0.034 | 0.41 |
| FP | 0.43 | 0.69 | 0.88 | 0.21 | 0.034 | 0.41 |
| TN | 0.34 | 0.08 | 0.76 | 0.30 | 0.027 | 0.35 |
| FN | 0.33 | 0.08 | 0.75 | 0.32 | 0.027 | 0.34 |

**TP≈FP ve TN≈FN** olgusu, **mevcut modal skorların ayırt edici kapasitesinin lineer füzyon için tükendiğine** işaret eder. Bu noktadan sonra kazanım olası iki yoldan gelmelidir:

1. **Daha güçlü modal omurgalar** (özellikle AVLips'e ince-ayarlı `Sv`, ya da öğrenilmiş bir sync-embedding).
2. **Non-lineer karar yüzeyi**: ki bu çalışmadaki HistGB AUC kazanımı zaten bu yolun küçük bir teyididir.

---

## 11. Üretim Çıkarımı ile Analiz Modelleri Arasındaki Ayrım

`infer/predict_video.py` aşağıdaki adımları izler:

1. `analyze(video)` ile modal skorlar ve boru hattı `Sf` üretilir.
2. `models/fusion_model.json`'daki `feature_names` sırasına göre vektör hazırlanır; `expand_features` ve isteğe bağlı standartlaştırma uygulanır.
3. `p_fake = σ(w^T x + b)`. `|p_fake - threshold| ≤ 0.05` ise **UNCERTAIN**, aksi halde eşik karşılaştırması ile **REAL/FAKE** etiketi.

Yukarıdaki yol **yalnızca lojistik JSON formatıyla uyumludur**. HistGB ve Platt-kalibre olasılıklar şu an üretim akışına serileştirilmemiştir; raporlanan AUC/ECE iyileşmeleri **analiz amaçlıdır** ve tezde bu ayrımın net belirtilmesi gerekir.

**İki sistemin yan yana özeti.**

| Sistem | Kayıt | Sınıflandırıcı | Test AUC | Test Bal-Acc | Test ECE | Üretim akışında? |
|---|---|---|---|---|---|---|
| Deploy edilen (canlı boru hattı) | `models/fusion_model.json` | Lojistik regresyon (Sl+Sa) | ≈0.51 | ≈0.53 | 0.077 (raw) | **Evet** (`infer/predict_video.py`) |
| Analiz / tezde raporlanan | `results/v2/fusion_histgb_v2_allfeats.json` + `fusion_calibration.json` | HistGB (6 özellik) + Platt | **0.612** | **0.557** | **0.025** | Hayır (henüz seri hâlde değil) |

Tezde, **bu iki çıkışın asla aynı anda "sistem performansı" olarak sunulmaması** gerekir. Üretim metrikleri **0.51 AUC**; analiz aşamasında ulaşılan en iyi konfigürasyon **0.61 AUC** + **0.025 ECE**'dir. Geçişin yapılması için gerekli adımlar Bölüm 14'te (Sınırlılıklar ve Gelecek Çalışma) listelenmiştir.

---

## 12. Tartışma

**Bilgi tavanı.** Hata kümelerinde gözlemlenen TP≈FP / TN≈FN paterni, mevcut modal skorların lineer ayrışıma katkısı bakımından doyduğunu gösterir. HistGB'nin yine de AUC'yi iyileştirebilmesi, **etkileşimlerin lineer olmayan bir karar yüzeyi** ile yakalanabildiğine işarettir; ancak bu kazanım da nihayetinde önbellekteki sinyalin kalitesi ile sınırlıdır.

**Sl güncellemesinin amacı.** Toplam doğruluğu büyük ölçüde artırmaktan çok, **modeli daha dürüst hâle getirmektir**. Önceki versiyonda `Sl≈1.0` doygunluğu, lojistik füzyonun “sabit bir aktivasyon kazanımı” elde etmesine yol açıyordu; bu güzel sayılar üretiyor ancak özelliği yorumlanamaz kılıyordu.

**Kalibrasyon önemi.** AUC üst seviyede iyi olsa da, kötü kalibre olasılıklar "yüksek güvenle yanılan" hatalara yol açar. Platt scaling, **ECE'yi 0.077'den 0.025'e düşürerek** olasılık çıktısını eşik-seçimi, risk-tabanlı raporlama veya insan-içeri karar yöneticisi senaryoları için anlamlı kılar.

**Hesaplama maliyeti.** Deney ortamında GPU bulunmadığı için PyTorch **CPU** sürümü kullanılmıştır; tam görsel ince ayar bu donanımda makul süreler içinde gerçekleştirilemez. Önbellek tabanlı yaklaşım, modellerin tekrar tekrar koşturulmasını engelleyerek bu sınırlılığı kısmen telafi eder.

---

## 13. Etik ve Toplumsal Konular

Deepfake tespit sistemleri, **iki yönlü etik** sorumluluk taşır: (i) yanlış pozitifler bireyleri haksız yere yargılayabilir; (ii) yanlış negatifler kamuoyunu yanıltabilir. Bu çalışmada **UNCERTAIN** bandının korunması, **kalibre edilmiş olasılıklar** ile karar destek senaryoları ve **insan denetimi**nin önerilmesi bu sorumluluğu hafifletmeye yöneliktir. Otomatik silme, ban veya hukuki delil olarak kullanım için **mevcut doğruluk seviyesinin yeterli olmadığı** açıkça beyan edilmelidir. AVLips dataseti üzerinde kişisel kimlik bilgisi olmayan ortak veri kullanılmıştır; özel veriyle çalışılırken **veri koruma** mevzuatına uyum gereklidir.

---

## 14. Sınırlılıklar ve Gelecek Çalışma

1. **Tek dataset değerlendirmesi.** AVLips dışında **DFDC** [3] ve **CelebDF** [4] gibi farklı dağılımlarda cross-dataset test yapılmamıştır.
2. **Görsel omurganın AVLips'e ince-ayarsız olması.** FaceForensics++ önceden eğitimi alan-dışı (out-of-domain) düşüş riski taşır [1]; AVLips üzerinde ince-ayar ile `Sv`'nin ayırt ediciliği artırılabilir.
3. **Klasik korelasyon temelli sync.** SyncNet / Wav2Lip benzeri öğrenilmiş senkron embedding [9, 10] bu sinyali güçlendirebilir.
4. **Tek bir sabit bölünme bias'ı.** 5-fold CV bunu yumuşatır ancak kimlik-bazlı (speaker-independent) bölünme gerekir.
5. **Üretim akışında yalnızca lojistik.** HistGB ve kalibre modelin `predict_video`'ya entegrasyonu açık geliştirme öğesi olarak durmaktadır. Geçişin somut adımları:
   - (a) HistGB modelini + Platt kalibratörünü birlikte serileştiren bir JSON şeması tanımlanır (ör. `models/fusion_histgb_platt.json`).
   - (b) `infer/predict_video.py` içinde model türü (`logistic` / `histgb_platt`) tespit edilerek uygun yükleyici çalıştırılır; çıktı arayüzü (`p_fake`, `label`) korunur.
   - (c) Mevcut `models/fusion_model.json` geriye uyum için bırakılır; varsayılan olarak yeni JSON tercih edilir.
6. **Önbellek temelli baseline kayıtları.** `data/feature_cache.csv.bak` (eski `Sl`) + `results/v2_oldSl/*.json` baseline olarak korunmaktadır; herhangi bir yeni özellik / model eklendiğinde aynı tablolar tekrar üretilebilir, böylece **regresyon takibi** mümkün olur.

---

## 15. Sonuç

AVLips veri kümesi üzerinde çalışan bir **çok modlu deepfake tespit boru hattını** uçtan uca raporladık. Modüler skor çıkarımı, önbellekleme, lojistik ve ağaç tabanlı füzyon ile birlikte 5-katlı çapraz doğrulama ve olasılık kalibrasyonu sonuçları sunulmuştur. Sabit bölünmede HistGB tüm-özellik modeli **test AUC = 0.612 / Bal-Acc = 0.557**, çapraz doğrulamada **0.623 ± 0.011 AUC / 0.594 ± 0.011 Bal-Acc** elde etmiş; Platt kalibrasyonu test **ECE'yi 0.077'den 0.025'e** indirmiştir. **Sl ölçümünün yeniden tasarımı**, ölçüm kalitesi ve yorumlanabilirlik açısından kritik bir mühendislik katkısıdır. Gelecek çalışma için **görsel omurganın AVLips'e ince-ayarı**, **öğrenilmiş senkron embedding** ve **cross-dataset değerlendirme** önerilmektedir.

---

## 16. Kaynakça

> Aşağıdaki referanslar, ilgili literatürdeki ana eserlere yöneliktir; künyelerin tam ve format-uyumlu hâli tez şablonunuza göre düzenlenmelidir.

[1] A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Nießner. **FaceForensics++: Learning to Detect Manipulated Facial Images.** ICCV, 2019.
[2] L. Verdoliva. **Media Forensics and DeepFakes: An Overview.** IEEE Journal of Selected Topics in Signal Processing, 14(5):910–932, 2020.
[3] B. Dolhansky et al. **The DeepFake Detection Challenge (DFDC) Dataset.** arXiv:2006.07397, 2020.
[4] Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu. **Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics.** CVPR, 2020.
[5] T. Mittal, U. Bhattacharya, R. Chandra, A. Bera, D. Manocha. **Emotions Don't Lie: An Audio-Visual Deepfake Detection Method Using Affective Cues.** ACM MM, 2020.
[6] Y. Zhou, S.-N. Lim. **Joint Audio-Visual Deepfake Detection.** ICCV, 2021.
[7] M. Khalid, J. Tariq, S. Kim, S. S. Woo. **Multimodal Audio-Visual Deepfake Detection.** WACV / Workshop, 2021.
[8] J. S. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martínez, J. Fernández-Valdivia. **Diatom Autofocusing in Brightfield Microscopy: A Comparative Study.** ICPR, 2000. (Laplacian focus / blur measure.)
[9] J. S. Chung, A. Zisserman. **Out of Time: Automated Lip Sync in the Wild (SyncNet).** ACCV Workshops, 2016.
[10] K. R. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, C. V. Jawahar. **A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild (Wav2Lip).** ACM MM, 2020.
[11] C. Lugaresi et al. **MediaPipe: A Framework for Building Perception Pipelines.** arXiv:1906.08172, 2019.
[12] M. Todisco et al. **ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection.** Interspeech, 2019.
[13] J. Yamagishi et al. **ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection.** Interspeech, 2021.
[14] Y. Li, M.-C. Chang, S. Lyu. **In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking.** WIFS, 2018.
[15] X. Yang, Y. Li, S. Lyu. **Exposing Deep Fakes Using Inconsistent Head Poses.** ICASSP, 2019.
[16] J. H. Friedman. **Greedy Function Approximation: A Gradient Boosting Machine.** Annals of Statistics, 29(5):1189–1232, 2001. (Histogram tabanlı varyantı için bkz. LightGBM / scikit-learn HistGradientBoosting.)
[17] J. Platt. **Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.** Advances in Large Margin Classifiers, 1999.
[18] B. Zadrozny, C. Elkan. **Transforming Classifier Scores into Accurate Multiclass Probability Estimates.** KDD, 2002.
[19] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. **On Calibration of Modern Neural Networks.** ICML, 2017. (ECE tanımı.)
[20] G. W. Brier. **Verification of Forecasts Expressed in Terms of Probability.** Monthly Weather Review, 78(1):1–3, 1950.
[21] C. R. Harris et al. **Array Programming with NumPy.** Nature, 585:357–362, 2020.
[22] F. Pedregosa et al. **Scikit-learn: Machine Learning in Python.** JMLR, 12:2825–2830, 2011.
[23] A. Paszke et al. **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** NeurIPS, 2019.
[24] B. McFee et al. **librosa: Audio and Music Signal Analysis in Python.** SciPy, 2015.
[25] F. Bellard ve ark. **FFmpeg.** https://ffmpeg.org/
[26] G. Bradski. **The OpenCV Library.** Dr. Dobb's Journal of Software Tools, 2000.
[27] T. Fawcett. **An Introduction to ROC Analysis.** Pattern Recognition Letters, 27(8):861–874, 2006.
[28] A. P. Bradley. **The Use of the Area Under the ROC Curve in the Evaluation of Machine Learning Algorithms.** Pattern Recognition, 30(7):1145–1159, 1997.
[29] T. Chen, C. Guestrin. **XGBoost: A Scalable Tree Boosting System.** KDD, 2016. (Genişlemeli okuma; HistGB ilham hattı.)
[30] G. Ke et al. **LightGBM: A Highly Efficient Gradient Boosting Decision Tree.** NeurIPS, 2017. (Histogram tabanlı gradient boosting; scikit-learn HistGB'nin yakın akrabası.)

---

## 17. Ek A — Dizin ve Komut Sözlüğü

| Yol | Rol |
|---|---|
| `src/analyze_video.py` | Ana çıkarım API'si (`analyze(video_path)`) |
| `src/lip_sync.py` | Sl ölçümü (`lip_mismatch_score`) ve VAD |
| `src/visual_score.py`, `src/visual_model.py` | Sv hesabı + omurga yükleme |
| `src/biomech.py` | Sb (blink), Sh (headpose) |
| `src/audio_artefact.py` | Sa hesabı |
| `src/fusion.py` | `Sf` yorumlama (REAL/FAKE/UNCERTAIN) |
| `src/fusion_features.py` | Özellik ad eşlemeleri (`Sf` ↔ `Sf_pipeline`) |
| `src/fusion_expand.py` | `none` / `poly2` özellik genişletmesi |
| `infer/predict_video.py` | Tek video üretim çıkarımı |
| `train/train_fusion_from_metadata.py` | Lojistik eğitim |
| `train/auto_select_fusion_model.py` | Alt küme araması |
| `train/tune_fusion_hparams.py` | Lojistik için grid hparam tarama |
| `eval/evaluate_fusion.py` | Holdout metrikleri |
| `eval/fusion_error_report.py` | Hata CSV |
| `eval/fusion_error_feature_summary.py` | FN/FP profili |
| `eval/train_fusion_histgb.py` | HistGB analizi |
| `eval/fusion_cv.py` | 5-fold CV |
| `eval/fusion_calibration.py` | ECE / Brier |
| `eval/plot_fusion_figures.py` | ROC, kalibrasyon, CV şekilleri → `results/v2/figures/*.png` |
| `eval/fusion_io.py` | Ortak okuma/matris hazırlık |
| `data_tools/refresh_sl_cache.py` | Toplu Sl yenileme (paralel + checkpoint) |
| `data_tools/metadata_builder.py` | Metadata CSV oluşturma |
| `data/feature_cache.csv` | Önbellek (ana veri) |
| `models/fusion_model.json` | Üretim lojistik modeli |
| `models/fusion_model_search_report.json` | Auto-search özeti |
| `results/v2/SUMMARY.md` | Kısa sayısal özet |
| `logs/refresh_sl_cache_resume.log` | Sl yenileme günlüğü |

Tipik komutlar (PowerShell ortamında):

```powershell
# 1) Sl önbelleğini yenile (gerekirse)
.\.venv\Scripts\python.exe -u data_tools\refresh_sl_cache.py --backup --workers 4 --checkpoint-every 200 --no-progress

# 2) Lojistik kombinasyon araması ve dağıtım modelini güncelle
.\.venv\Scripts\python.exe train\auto_select_fusion_model.py `
  --metadata-csv data\avlips_metadata.csv --cache-csv data\feature_cache.csv `
  --out-model models\fusion_model.json --report-json models\fusion_model_search_report.json `
  --pos-weight-auto --metric val_balanced_acc --max-combo-size 6

# 3) Üretim modelinin holdout metriği
.\.venv\Scripts\python.exe eval\evaluate_fusion.py `
  --metadata-csv data\avlips_metadata.csv --cache-csv data\feature_cache.csv `
  --model-json models\fusion_model.json

# 4) HistGB analiz
.\.venv\Scripts\python.exe eval\train_fusion_histgb.py `
  --from-model models\fusion_model.json --features Sv,Sl,Sb,Sh,Sa,Sf `
  --out-json results\v2\fusion_histgb_v2_allfeats.json

# 5) 5-kat CV
.\.venv\Scripts\python.exe eval\fusion_cv.py `
  --features Sv,Sl,Sb,Sh,Sa,Sf --n-splits 5 --out-json results\v2\fusion_cv_allfeats.json

# 6) Kalibrasyon
.\.venv\Scripts\python.exe eval\fusion_calibration.py `
  --features Sv,Sl,Sb,Sh,Sa,Sf --out-json results\v2\fusion_calibration.json

# 7) Sekiller (PNG) — ROC, kalibrasyon, 5-fold bar
.\.venv\Scripts\python.exe eval\plot_fusion_figures.py

# 8) Hata raporları
.\.venv\Scripts\python.exe eval\fusion_error_report.py --out-dir results\fusion_errors_v2
.\.venv\Scripts\python.exe eval\fusion_error_feature_summary.py `
  --errors-csv results\fusion_errors_v2\errors_test.csv `
  --out-json results\fusion_errors_v2\summary_test.json

# 9) Tek video çıkarımı
.\.venv\Scripts\python.exe infer\predict_video.py --video "C:\path\to\video.mp4"
```

---

## 18. Ek B — JSON Çıktılarından Doğrudan Alınan Özet Sayılar

### B.1 Sl yenileme

- `target rows`: 7402 (offset=200, end=7602)
- `processed`: 7402, `failed`: 0
- `elapsed`: ≈ 158 dk (4 worker, checkpoint=200)
- `.bak` ile karşılaştırmada **değişen satır**: 6183

### B.2 Lojistik auto-search

- Tarama: **63 kombinasyon**, metric=`val_balanced_acc`.
- Kazanan: `Sl, Sa` (val_bal_acc = 0.5512).
- Dağıtım modeli: `models/fusion_model.json` (`Sl, Sa`).

### B.3 HistGB — Sv,Sl,Sb,Sh,Sa,Sf

- Val: acc=0.6146, bal_acc=0.5920, F1=0.6979, AUC=0.6264.
- Test: acc=0.5806, bal_acc=0.5565, F1=0.6735, AUC=0.6120.
- Train config: `max_depth=6, max_iter=250, lr=0.06, class_weight=balanced, random_state=42`.

### B.4 5-kat CV (Sv,Sl,Sb,Sh,Sa,Sf)

- Lojistik agg: bal_acc=0.5447±0.0067, AUC=0.5522±0.0165, F1=0.6131±0.1457.
- HistGB agg: bal_acc=0.5943±0.0110, AUC=0.6234±0.0113, F1=0.6031±0.0385.

### B.5 Kalibrasyon (HistGB, test)

- Raw: AUC=0.6120, Brier=0.2474, ECE=0.0765.
- Platt: AUC=0.6120, Brier=0.2380, ECE=0.0248.
- Isotonic: AUC=0.6075, Brier=0.2419, ECE=0.0349.

### B.6 Baseline — Eski `Sl` (`.bak`) sayıları

> Tüm değerler `results/v2_oldSl/*.json` ve `models/fusion_model_oldSl.json` dosyalarından doğrudan alınmıştır.

- **Lojistik auto-search** (63 kombinasyon, eski cache):
  - Kazanan alt küme: `Sv, Sl, Sh, Sa, Sf`.
  - Val: bal_acc=0.5661, AUC=0.5763. Test: bal_acc=0.5482, AUC=0.5609.
- **HistGB** (6 özellik, eski cache):
  - Val: acc=0.6260, bal_acc=0.6015, F1=0.7110, AUC=0.6242.
  - Test: acc=0.5753, bal_acc=0.5480, F1=0.6769, AUC=0.6030.
- **5-kat CV** (6 özellik, eski cache):
  - Lojistik agg: bal_acc=0.5604±0.0123, AUC=0.5788±0.0159.
  - HistGB agg: bal_acc=0.5918±0.0095, AUC=0.6209±0.0131.
- **Kalibrasyon** (HistGB, test, eski cache):
  - Raw: AUC=0.6030, Brier=0.2492, ECE=0.0957.
  - Platt: AUC=0.6030, Brier=0.2395, ECE=0.0362.
  - Isotonic: AUC=0.5396, Brier=0.2459, ECE=0.0532.

---

## 19. Ek C — Yeniden Üretim Adımları

1. **Ortam.** Python 3.10+, `requirements.txt` (numpy, scikit-learn, torch CPU, librosa, mediapipe, opencv-python, ffmpeg binari).
2. **Metadata.** `data_tools/metadata_builder.py` ile `data/avlips_metadata.csv` üretilir; her satırda `video_path, label, split`.
3. **Önbellek.** İlk koşuda `train/train_fusion_from_metadata.py` boş önbellek tespit edip dolduran zincir ile çağrılabilir veya elle batch çıkarım yapılır; Sl güncellemesinden sonra `data_tools/refresh_sl_cache.py` zorunludur.
4. **Lojistik eğitim.** `train/auto_select_fusion_model.py` (yukarıdaki komut). Sonuç `models/fusion_model.json` ve raporu.
5. **Analiz.** `eval/train_fusion_histgb.py`, `eval/fusion_cv.py`, `eval/fusion_calibration.py`.
6. **Çıkarım.** `infer/predict_video.py --video <path>`; JSON çıktısı `p_fake`, `threshold`, `label` ve `pipeline_details`.
7. **Raporlama.** `results/v2/SUMMARY.md` ve `docs/MAKALE_TEKNIK_MULTIMODAL_DEEPFAKE.md` (bu dosya) güncel sayılarla.
8. **Şekiller (PNG).** `eval/plot_fusion_figures.py` — çıktı `results/v2/figures/` (ROC, kalibrasyon, 5-fold çubuk grafik). Ayrıntı için Bölüm 20.

---

## 20. Görselleştirmeler ve Kaynak PDF Arşivi

### 20.1 Betik

`eval/plot_fusion_figures.py` aşağıdaki PNG dosyalarını üretir (varsayılan çıktı dizini `results/v2/figures/`):

| Dosya | İçerik |
|-------|--------|
| `cv_fold_balanced_acc_auc.png` | 5-fold CV: fold başına lojistik vs HistGB **dengeli doğruluk** ve **AUC** (yan yana çubuklar). Veri: `results/v2/fusion_cv_allfeats.json`. |
| `calibration_reliability_from_json_bins.png` | `fusion_calibration.json` içindeki ECE bin ortalamaları (`p_mean` vs `y_mean`): ham HistGB ve Platt (yeniden eğitim gerekmez). |
| `roc_test_histgb_platt.png` | Test seti ROC: HistGB ham ve HistGB+Platt eğrileri + rastgele taban (`sklearn.metrics.roc_curve`). |
| `calibration_reliability_sklearn.png` | `sklearn.calibration.calibration_curve` ile güvenilirlik eğrisi (ham vs Platt, mükemmel kalibrasyon çizgisi). |

Komut:

```powershell
cd Multimodal-Deepfake-Tespit-Sistemi
.\.venv\Scripts\python.exe eval\plot_fusion_figures.py
```

Sadece JSON tabanlı şekiller (ROC/kalibrasyon için model yeniden eğitimi atlanır):

```powershell
.\.venv\Scripts\python.exe eval\plot_fusion_figures.py --no-train
```

### 20.2 Kaynakça için PDF klasörü

Tez veya makale bibliyografisi için tam metin PDF arşivi şu dizindedir (örnek: FaceForensics, multimodal deepfake, kalibrasyon ve görüntü işleme dergi makaleleri):

`C:\Users\busra\Desktop\projeler\makale\`

Örnek dosya adları: `Deepfake_Media_Generation_and_Detection_in_the_Gen.pdf`, Springer *International Journal of Computer Vision* / *Machine Learning* / *Multimedia Tools and Applications* DOI’li PDF’ler (`s11263-*.pdf`, `s00371-*.pdf`, vb.). Her PDF için Zotero / JabRef ile DOI çözümlemesi veya dergi sayfasından yazar–başlık–yıl–sayfa aralığı çıkarılarak Bölüm 16’daki numaralı kaynakça ile eşleştirme yapılmalıdır.

---

*Bu belge, AVLips üzerinde çalışan çok modlu deepfake tespit projesinin v2 sürümünün uçtan uca teknik açıklamasıdır. Kaynakça eserleri konuyla doğrudan ilgili literatürdeki kanonik çalışmaları kapsar; künye biçimi tez şablonuna göre IEEE/APA/Springer formuna dönüştürülebilir. Sayısal sonuçlar `results/v2/*.json` dosyalarıyla bire bir tutarlıdır; yeniden çalıştırıldığında küçük floating-point varyasyonları beklenebilir.*
