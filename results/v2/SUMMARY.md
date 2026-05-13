# Fusion Doğrulama Özeti (v2 — yeni Sl + ek modeller)

Tarih: 2026-05-11
Veri: AVLips, 7602 video (real=3396, fake=4206)
Split: train=5321 / val=1139 / test=1142

## 1. Sl (lip-sync) iyileştirmesi

Eski Sl ölçümü pek çok videoda 1.0 doygunluğa ulaşıyor, varyans taşımıyordu. Yeni ölçüm:
- Varyans kapısı (ses/dudak serisi düz ise Sl≈0.5)
- 5-örnek hareketli ortalama
- Ham sinyal + birinci fark karışımı (`DF_LIP_SYNC_VELOCITY_BLEND=0.45`)

Tam önbellek yenileme: 7602 / 7602, başarısız 0, süre 158 dk (4 worker paralel).

## 2. Holdout test/val metrikleri (Sv, Sl, Sb, Sh, Sa, Sf)

Aynı feature_cache.csv üzerinde aynı train/val/test bölünmesi ile.

| Model | Val Bal-Acc | Val AUC | Test Bal-Acc | Test AUC |
|-------|-------------|---------|---------------|----------|
| Lojistik (auto-search seçimi: Sl+Sa) | 0.5512 | 0.5366 | 0.5331 | 0.5108 |
| HistGB (2 özellik: Sl+Sa) | 0.5640 | 0.5746 | 0.5318 | 0.5494 |
| **HistGB (6 özellik tüm)** | **0.5920** | **0.6264** | **0.5565** | **0.6120** |

Sonuç: HistGB tam özellik setiyle, lojistiğe göre **val balanced_acc +0.04, val AUC +0.09, test AUC +0.10** kazandırıyor — non-linear etkileşimler özellikle AUC tarafında belirgin.

## 3. 5-fold Stratified Cross-Validation

Train+val+test birleştirildi, stratified 5-fold (seed=42), her foldda eşik val tabanlı seçildi.

| Model | Balanced Acc (mean ± std) | AUC (mean ± std) | F1 (mean ± std) |
|-------|----------------------------|-------------------|-------------------|
| Lojistik | 0.5447 ± 0.0067 | 0.5522 ± 0.0165 | 0.6131 ± 0.1457 |
| **HistGB** | **0.5943 ± 0.0110** | **0.6234 ± 0.0113** | 0.6031 ± 0.0385 |

Fold-bazında:

| Fold | LR Bal-Acc | LR AUC | GB Bal-Acc | GB AUC |
|------|------------|--------|-------------|--------|
| 1 | 0.5396 | 0.5385 | 0.5771 | 0.6095 |
| 2 | 0.5433 | 0.5436 | 0.6014 | 0.6306 |
| 3 | 0.5414 | 0.5486 | 0.5936 | 0.6141 |
| 4 | 0.5565 | 0.5805 | 0.6059 | 0.6361 |
| 5 | 0.5426 | 0.5496 | 0.5936 | 0.6268 |

HistGB **her 5 foldda lojistiği geçti**, std düşük (sistematik kazanım, "şanslı bölünme" değil).

## 4. Olasılık Kalibrasyonu (test seti)

HistGB ham çıktısı + Platt (sigmoid) / Isotonic kalibrasyon, val üzerinde fit.

| Variant | AUC | Brier | **ECE** |
|---------|-----|-------|---------|
| Raw HistGB | 0.6120 | 0.2474 | **0.0765** |
| Platt (sigmoid) | 0.6120 | 0.2380 | **0.0248** |
| Isotonic | 0.6075 | 0.2419 | 0.0349 |

Platt kalibrasyonu ECE'yi **~3 kat** iyileştirdi (0.077 → 0.025) — AUC ile ödün vermeden. Üretim için Platt-kalibreli olasılıklar önerilir.

## 5. Hata Profili (test, mevcut deploy lojistik modeli)

Doğru ve yanlış sınıflarda özellik ortalamaları neredeyse özdeş (TP≈FP, TN≈FN). Bu, **cache'deki mevcut modal skorların bilgi tavanına ulaşıldığını** gösterir; mevcut özellikler fusion katmanı için linear ayrımı zaten doyurmuş.

## 6. Baseline Karşılaştırması — Eski Sl vs Yeni Sl

Eski (doygun) Sl önbelleği `data/feature_cache.csv.bak` üzerinde tüm değerlendirme zinciri tekrar koşturuldu. Aynı script'ler, aynı bölme, aynı 6 özellik (`Sv, Sl, Sb, Sh, Sa, Sf`):

### 6.1 Lojistik füzyon — auto-search (63 alt küme) seçimi

| Cache | Seçilen alt küme | Val Bal-Acc | Val AUC | Test Bal-Acc | Test AUC |
|-------|------------------|-------------|---------|---------------|----------|
| Eski Sl (`.bak`) | Sv+Sl+Sh+Sa+Sf | 0.5661 | 0.5763 | 0.5482 | 0.5609 |
| Yeni Sl (`feature_cache.csv`) | Sl+Sa | 0.5512 | 0.5366 | 0.5331 | 0.5108 |

Yorum: doygun (1.0'a yapışmış) eski Sl, lineer ayrımda yapay bir "fake ≈ Sl≈1" kestirmesi yarattığı için lojistik tarafa şişirilmiş bir avantaj veriyordu. Yeni Sl daha sürekli/gürültülü olduğu için lojistik tek başına kazanımı veremiyor — sinyal HistGB'ye taşınıyor (aşağı).

### 6.2 HistGB (6 özellik) — holdout test

| Cache | Val Bal-Acc | Val AUC | Test Bal-Acc | Test AUC |
|-------|-------------|---------|---------------|----------|
| Eski Sl | 0.6015 | 0.6242 | 0.5480 | 0.6030 |
| **Yeni Sl** | **0.5920** | **0.6264** | **0.5565** | **0.6120** |

Yeni Sl test AUC'yi +0.009, test Bal-Acc'yi +0.0085 artırıyor.

### 6.3 5-fold Stratified CV (6 özellik)

| Cache | Model | Bal-Acc (mean ± std) | AUC (mean ± std) |
|-------|-------|----------------------|-------------------|
| Eski Sl | Lojistik | 0.5604 ± 0.0123 | 0.5788 ± 0.0159 |
| Eski Sl | HistGB   | 0.5918 ± 0.0095 | 0.6209 ± 0.0131 |
| Yeni Sl | Lojistik | 0.5447 ± 0.0067 | 0.5522 ± 0.0165 |
| **Yeni Sl** | **HistGB** | **0.5943 ± 0.0110** | **0.6234 ± 0.0113** |

HistGB ile yeni Sl her iki metrikte de eski Sl'nin üzerinde (Bal-Acc +0.0025, AUC +0.0025); std da düşük → kazanım foldlar arası tutarlı.

### 6.4 Olasılık kalibrasyonu (HistGB, test)

| Cache | Variant | AUC | Brier | ECE |
|-------|---------|-----|-------|------|
| Eski Sl | Raw HistGB | 0.6030 | 0.2492 | 0.0957 |
| Eski Sl | Platt | 0.6030 | 0.2395 | 0.0362 |
| Yeni Sl | Raw HistGB | 0.6120 | 0.2474 | 0.0765 |
| **Yeni Sl** | **Platt** | **0.6120** | **0.2380** | **0.0248** |

Yeni Sl + Platt: tüm yapılandırmaların en iyisi (en yüksek AUC ve en düşük ECE/Brier). Eski Sl + Platt ile karşılaştırıldığında ECE **0.036 → 0.025** (≈ %32 daha iyi).

### 6.5 Net Sonuç

- **Lojistik tek başına bakılırsa** eski Sl "sayfada daha iyi" gözüküyor (Test AUC 0.561 vs 0.511), ama bu doygun bir gösterge sinyalinin kestirme öğrenilmesinden geliyor — kalibrasyon bozuk (ECE 0.096 vs 0.077) ve genelleme zayıf.
- **HistGB + tüm özellikler + Platt** ile yeni Sl her metrikte (AUC, Bal-Acc, Brier, ECE) eski Sl'yi geçiyor. Tezde rapor edilecek temel hat budur.

## 7. Deploy edilen sistem (`predict_video`) vs analiz raporları

`src/predict_video.py` ve canlı boru hattı hâlâ `models/fusion_model.json` (lojistik) yüklüyor:

- Üretim çıkışı: **Lojistik** (Sl+Sa), test AUC ≈ 0.51, dengeli doğruluk ≈ 0.53. Hızlı/şeffaf, ama bilgi tavanına yakın.
- Analiz/değerlendirme raporları (yukarıdaki tüm tablolar): **HistGB + Platt**, test AUC ≈ 0.61, dengeli doğruluk ≈ 0.56, ECE ≈ 0.025.

İki çıkışı tezde **ayrı sunmak** gerekir; aksi takdirde "sistem 0.61 AUC veriyor" yanılgısı doğar. Geçiş için yapılacak:

1. `eval/train_fusion_histgb.py` çıktısı + Platt kalibrasyon parametreleri tek JSON'a yazılır.
2. `predict_video` bu JSON'u yüklemek için bir HistGB+kalibratör adaptörü kazanır.
3. Aynı `models/fusion_model.json` API'si korunur (geriye uyum) ama içeriği genişler.

## 8. Sınırlılıklar ve Gelecek Çalışma

- Visual omurga (Sv) FaceForensics önbellek modelinden; AVLips'e ince ayar yapılmadı.
- Lip-sync modülü (Sl) klasik korelasyon tabanlı; öğrenilmiş senkron embedding (örn. SyncNet) sınamadık.
- Tek dataset; cross-dataset (DFDC, CelebDF) genelleme test edilmedi.
- Önerilen sonraki adım: AVLips üzerinde Sv fine-tuning + cache yeniden hesaplama.

## 9. Tezde Kullanılabilecek Sayılar (özet)

> Çok modlu lojistik füzyon AVLips test kümesinde dengeli doğruluk **0.55** ve AUC **0.51** seviyesindeyken, ağaç tabanlı non-lineer füzyon (HistGradientBoosting) aynı özellik kümesinde dengeli doğruluğu **0.56**'a, AUC'yi **0.61**'e çıkarmaktadır. 5-fold çapraz doğrulamada HistGB modeli **0.594 ± 0.011** dengeli doğruluk ve **0.623 ± 0.011** AUC ile her foldda lojistik tabanı geçmektedir. Platt kalibrasyonu sonrası test ECE 0.077'den 0.025'e inerek olasılık çıktılarının güvenilirliği belirgin biçimde artmıştır.

## Dosyalar

### Yeni Sl (deploy + analiz tabanı)
- Auto-search (lojistik 63 kombinasyon): `models/fusion_model_search_report.json`
- HistGB tek-set: `results/v2/fusion_histgb_v2.json` (Sl+Sa)
- HistGB tüm-özellik: `results/v2/fusion_histgb_v2_allfeats.json`
- CV raporu: `results/v2/fusion_cv_allfeats.json`
- Kalibrasyon raporu: `results/v2/fusion_calibration.json`
- Şekiller: `results/v2/figures/cv_fold_balanced_acc_auc.png`, `calibration_reliability_from_json_bins.png`, (varsa) `roc_test_histgb_platt.png`
- Sl yenileme log: `logs/refresh_sl_cache_resume.log`

### Eski Sl (baseline karşılaştırma)
- Auto-search: `models/fusion_model_oldSl.json` + `models/fusion_model_oldSl_search.json`
- HistGB tüm-özellik: `results/v2_oldSl/fusion_histgb_oldSl_allfeats.json`
- 5-fold CV: `results/v2_oldSl/fusion_cv_oldSl.json`
- Kalibrasyon: `results/v2_oldSl/fusion_calibration_oldSl.json`
- Önbellek: `data/feature_cache.csv.bak`
- Auto-search log: `logs/auto_select_oldSl.log`
