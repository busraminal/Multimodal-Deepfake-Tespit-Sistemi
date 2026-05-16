# Poster grafik açıklamaları

## Güvenilirlik eğrisi (`results/v2/figures/calibration_reliability_sklearn.png`)

**Ne ölçülür?** Modelin verdiği sahte olasılığı (`p_fake`, 0–1) ile test setinde gerçekten sahte çıkan örnek oranının uyumu.

**Eksenler:** X = bin ortalama tahmin olasılığı; Y = gözlenen pozitif (sahte) oranı. Kesik çizgi = mükemmel kalibrasyon.

**Ham HistGB (mavi):** Ham olasılıklar çizgiden sapar; düşük bölgede yetersiz güven, yüksek bölgede aşırı güven görülebilir.

**Platt / sigmoid (turuncu):** Doğrulama setinde Platt ölçeklemesi sonrası eğri çapraza yaklaşır. Test **ECE** ~0.077 → ~0.025; **AUC (~0.61)** korunur.

**Poster cümlesi:** *HistGB füzyonunda Platt kalibrasyonu, test ECE’yi 0.077’den 0.025’e indirerek olasılık çıktılarını ayırt edici gücü koruyarak güvenilir hale getirmiştir.*

## ROC (`results/v2/figures/roc_test_histgb_platt.png`)

Test setinde HistGB ham ve Platt olasılıkları; AUC ~0.612. Platt kalibrasyonu ROC alanını pratikte değiştirmez, güvenilirliği iyileştirir.

## 5-fold CV (`results/v2/figures/cv_fold_balanced_acc_auc.png`)

Stratified 5-fold; özellikler Sv, Sl, Sb, Sh, Sa, Sf. HistGB her foldda lojistik tabanı geçer (Bal-Acc ort. ~0.594, AUC ort. ~0.623).
