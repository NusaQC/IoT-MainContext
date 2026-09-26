# NusaQC — Rencana, Arsitektur, & Master Prompt AI Track
### COMPFEST 18 AI Innovation Challenge (AIC) · Smart Manufacturing Track
### Fokus Utama: Peningkatan Metrik Kualitas Model & Pemangkasan Latensi Edge

Folder ini merangkum seluruh strategi perbaikan AI NusaQC untuk babak Hackathon 10 Jam, termasuk pembagian beban komputasi menggunakan **2 Akun Kaggle GPU secara simultan (paralel)**, arsitektur *dual-engine*, penanganan data leakage, metode *data slicing*, serta master prompt siap pakai untuk kedua akun.

---

## 📁 Struktur Dokumen AI Track

| Dokumen | Topik & Peran Utama | Target Fokus |
| :--- | :--- | :--- |
| **[`01_KEBUTUHAN_DAN_ARSITEKTUR_AI.md`](./01_KEBUTUHAN_DAN_ARSITEKTUR_AI.md)** | • Arsitektur Dual-Engine: Model 1 Freshness (MobileNetV3-Small) & Model 2 Defect (YOLOv8s).<br>• Spesifikasi Dataset: DaFiF (2.536 citra), FFE (4.390 citra), Roboflow + Pseudo-Labeling.<br>• Mitigasi Data Leakage via *Grouped Stratified Split by Day/Session*.<br>• Metodologi Build-the-Eval Track (Bobot 20%). | Fondasi Teknis AI |
| **[`02_STRATEGI_PARALEL_2_KAGGLE_4_SESI.md`](./02_STRATEGI_PARALEL_2_KAGGLE_4_SESI.md)** | • Blueprint 4 Sesi GPU Paralel di 2 Akun Kaggle (A/B Testing Champion vs Challenger).<br>• Matriks perbandingan 4 eksperimen (Weighted Loss vs Focal Loss, Glare Aug vs INT8).<br>• Alur konvergensi pukul 14.30 WIB untuk memilih 2 model final terbaik. | Strategi Operasional Komputasi |
| **[`PROMPT_AKUN1_MODEL1_WEIGHTED_LOSS.md`](./PROMPT_AKUN1_MODEL1_WEIGHTED_LOSS.md)** | • **Akun 1 Sesi 1 (Model 1 Champion)**: Retraining MobileNetV3-Small dengan Cost-Sensitive Loss ($W=[1, 1.2, 5]$) untuk menekan Critical Escape Rate ke 0.00%. | Model 1: Freshness Engine |
| **[`PROMPT_AKUN1_MODEL2_GLARE_AUG.md`](./PROMPT_AKUN1_MODEL2_GLARE_AUG.md)** | • **Akun 1 Sesi 2 (Model 2 Champion)**: Retraining YOLOv8s dengan Specular Glare Augmentation (injeksi elips lendir putih) untuk menekan False Alarm ke <= 5.0%. | Model 2: Defect Engine |
| **[`PROMPT_AKUN2_MODEL1_FOCAL_LOSS.md`](./PROMPT_AKUN2_MODEL1_FOCAL_LOSS.md)** | • **Akun 2 Sesi 3 (Model 1 Challenger)**: Retraining MobileNetV3 dengan Multiclass Focal Loss ($\gamma=2.0$) untuk menguji resolusi ambiguitas Day 4. | Model 1: Studi Ablasi |
| **[`PROMPT_AKUN2_MODEL2_INT8_LATENCY.md`](./PROMPT_AKUN2_MODEL2_INT8_LATENCY.md)** | • **Akun 2 Sesi 4 (Model 2 Speedup)**: Dynamic INT8 Quantization ONNX Runtime & komparasi resolusi 416 vs 640 untuk memangkas latensi CPU RPi 4 ke ~1.000 ms. | Model 2: Mesin Kompresi |
| **[`03_TEST_SUITE_DAN_EVAL_TRACK.md`](./03_TEST_SUITE_DAN_EVAL_TRACK.md)** | • Metodologi data slicing (`nominal`, `low_light`, `specular_glare`, `safety_critical`).<br>• Kode lengkap modul `eval/slices.py`, `eval/metrics.py`, dan `eval/run_eval.py`.<br>• Master prompt penyusunan naskah akademik *Evaluation Artifact* PDF (20.30 WIB) & simulasi Q&A juri. | Deliverable Eval Track (20%) |

---

## 🎯 Target Akhir AI Track (Sebelum vs Sesudah)

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                RINGKASAN TARGET PERBAIKAN AI                         │
├────────────────────────────────┬───────────────────┬─────────────────────────────────┤
│ Aspek / Indikator Kinerja      │ Baseline Awal     │ Target Akhir Pasca-Iterasi      │
├────────────────────────────────┼───────────────────┼─────────────────────────────────┤
│ Model 1 Macro F1               │ 66.48%            │ ≥ 76.50% (+10.02%)              │
│ Critical Escape Rate (C -> A)  │ 2.54%             │ 0.00% (Zero-Escape Bencana FDA) │
│ Model 2 Defect mAP50           │ 0.682             │ ≥ 0.735 (pada model INT8)       │
│ False Alarm Glare Rate         │ 18.40%            │ ≤ 5.00% (Turun 3.7x lipat)      │
│ Latensi Model 2 (RPi 4 CPU)    │ 2.440 ms (FP32)   │ ~1.000 ms (INT8) [2.4x Speedup] │
│ Total Latensi Dual-AI RPi 4    │ 2.469 ms (FP32)   │ ~1.028 ms (INT8) [2.4x Speedup] │
│ Ukuran File Model ONNX YOLO    │ 43.0 MB           │ ~11.5 MB (-73% Hemat Memori)    │
└────────────────────────────────┴───────────────────┴─────────────────────────────────┘
```
