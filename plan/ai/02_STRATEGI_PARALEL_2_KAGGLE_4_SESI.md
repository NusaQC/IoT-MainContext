# NusaQC — Strategi Paralel 2 Akun Kaggle (4 Sesi GPU Simultan)
### Konsep: Eksperimen A/B Testing Kompetitif (Champion vs. Challenger)
### Target: Hackathon 10 Jam (Fase I & II: 09.00 – 15.00 WIB) · Bobot Eval Track: 20%

Dokumen ini memetakan pemanfaatan maksimal kuota komputasi cloud: **1 akun Kaggle dapat menjalankan 2 notebook GPU secara bersamaan**. Dengan **2 Akun Kaggle**, tim mengeksekusi **4 Sesi GPU Paralel** sekaligus untuk menguji dua hipotesis pengembangan bersaing (*Dual-Hypothesis Experimentation*).

---

## 1. Mengapa Strategi 4 Sesi Paralel Sangat Unggul?

Di Rulebook Babak Final Bagian 4 & 5, juri AI Engineer Fasilkom UI secara spesifik mencari:
> *"Apakah tim dapat menjelaskan alasan kegagalan eksperimen perbaikan tertentu (failed experiments) dan membuktikan kausalitas kenaikan metrik?"*

Jika tim hanya mencoba 1 cara lalu berhasil, juri menganggap itu kebetulan. Namun dengan **4 Sesi Paralel (A/B Testing)**:
1. **Model 1 Freshness:** Kita mengadu **Weighted Cross-Entropy** (Akun 1) vs **Focal Loss** (Akun 2).
2. **Model 2 Defect:** Kita mengadu **YOLOv8s Glare-Augmented FP32** (Akun 1) vs **YOLOv8s/n Dynamic INT8 Quantized** (Akun 2).
3. **Pada Pukul 14.30 WIB:** Hasil dari keempat sesi diuji pada test suite yang sama (`eval/run_eval.py`). 
   - Model dengan performa terbaik dipilih sebagai **Model Produksi Final** (*Champion*).
   - Model yang kalah TIDAK SIA-SIA, melainkan dijadikan bab khusus di PDF *Evaluation Artifact*: **"Analisis Kegagalan Hipotesis & Pembelajaran Eksperimen"** (*Challenger / Failed Experiment*). Ini menjamin poin maksimal dari dewan juri!

---

## 2. Pemetaan 4 Sesi GPU di 2 Akun Kaggle

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        AKUN KAGGLE 1: DATA-CENTRIC & LOSS TRACK                        │
├────────────────────────────────────────────┬───────────────────────────────────────────┤
│ SESI 1 (GPU T4)                            │ SESI 2 (GPU T4)                           │
│ • Model: Model 1 Freshness (MobileNetV3)   │ • Model: Model 2 Defect (YOLOv8s)         │
│ • Hipotesis: Cost-Sensitive Weighted Loss  │ • Hipotesis: Specular Glare Augmentation  │
│   (Penalti W_Grade_C = 5.0)                │   (Injeksi elips putih pantulan lendir)   │
│ • Target: Macro F1 >= 76.5%, Escape C = 0% │ • Target: Glare False Alarm <= 5.0%       │
│ • File Prompt:                             │ • File Prompt:                            │
│   `ai/PROMPT_AKUN1_MODEL1_WEIGHTED_LOSS.md`│   `ai/PROMPT_AKUN1_MODEL2_GLARE_AUG.md`   │
└────────────────────────────────────────────┴───────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     AKUN KAGGLE 2: ARCHITECTURE & COMPRESSION TRACK                    │
├────────────────────────────────────────────┬───────────────────────────────────────────┤
│ SESI 3 (GPU T4 / P100)                     │ SESI 4 (GPU T4 / CPU)                     │
│ • Model: Model 1 Freshness (MobileNetV3)   │ • Model: Model 2 Defect (YOLOv8s / v8n)   │
│ • Hipotesis: Focal Loss (gamma=2.0) &      │ • Hipotesis: Dynamic INT8 Quantization    │
│   Class Rebalancing Hard-Mining            │   & Input Resolution 416x416 Speedup      │
│ • Target: Mengatasi ambiguitas Day 4       │ • Target: Latensi RPi 4 CPU ~1.000 ms     │
│ • File Prompt:                             │ • File Prompt:                            │
│   `ai/PROMPT_AKUN2_MODEL1_FOCAL_LOSS.md`   │   `ai/PROMPT_AKUN2_MODEL2_INT8_LATENCY.md`│
└────────────────────────────────────────────┴───────────────────────────────────────────┘
```

---

## 3. Matriks Perbandingan 4 Eksperimen

| Sesi | Akun | Model Target | Hipotesis / Metode Perbaikan | Metrik yang Diuji | Peran di Hasil Akhir |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **Sesi 1** | Akun 1 | **Model 1** Freshness | **Cost-Sensitive Cross-Entropy** ($W = [1, 1.2, 5]$) | Zero Critical Escape Rate (Grade C $\rightarrow$ A) | Calon Utama Model 1 Final |
| **Sesi 2** | Akun 1 | **Model 2** Defect | **Specular Glare Augmentation** (OpenCV HSV elips) | False Alarm Glare Rate pada ikan basah | Calon Bobot Deteksi Terbaik |
| **Sesi 3** | Akun 2 | **Model 1** Freshness | **Focal Loss** ($\gamma=2.0, \alpha=0.25$) & Hard Mining | Resolusi ambiguitas kelas batas (Grade B) | Model Komparasi / Ablasi |
| **Sesi 4** | Akun 2 | **Model 2** Defect | **Dynamic INT8 Quantization** & Resolusi 416 vs 640 | Latensi Forward Pass CPU ARM Cortex-A72 | Mesin Kompresi Model 2 Final |

---

## 4. Alur Konvergensi Pukul 14.30 WIB (Menentukan 2 Model Final)

Meskipun berjalan 4 eksperimen, di akhir hackathon sistem NusaQC **tetap mengintegrasikan TEPAT 2 MODEL**:

1. **Pemilihan Model 1 (Pukul 14.30 WIB):**
   - Jalankan script evaluasi pada hasil **Sesi 1 (Weighted Loss)** dan **Sesi 3 (Focal Loss)** terhadap test slice `safety_critical`.
   - Jika Sesi 1 menghasilkan Escape Rate Grade C = 0.00% dan Macro F1 lebih tinggi $\rightarrow$ Sesi 1 dinobatkan sebagai pemenang: `AI/model-1/mobilenetv3_freshness.onnx`.
   - Hasil Sesi 3 dirangkum ke laporan: *"Mengapa Focal Loss kurang optimal dibanding Cost-Sensitive Loss pada data ikan"*.

2. **Pemilihan Model 2 (Pukul 14.40 WIB):**
   - Ambil bobot terlatih dari **Sesi 2 (Glare Augmentation)** yang memiliki mAP50 tertinggi dan bebas false alarm.
   - Masukkan bobot tersebut ke pipeline **Kuantisasi Dinamis INT8 dari Sesi 4**.
   - Hasilnya adalah satu model deteksi final yang **akurat (tahan kilau lendir) SEKALIGUS super kencang (~1.000 ms di CPU RPi 4)**: `AI/model-2/nusaqc_model2_defect_detector.onnx`.
   - Eksperimen resolusi 416x416 dari Sesi 4 dirangkum ke laporan: *"Trade-off latensi vs deteksi parasit kecil (< 20 px)"*.

3. **15.00 WIB (Siap Presentasi WIZ.AI & Checkpoint 2):**
   - Push Git Tag: `checkpoint-2-iteration`.
   - Tim memiliki data empiris 4 eksperimen sekaligus untuk dipaparkan secara saintifik kepada mentor WIZ.AI.
