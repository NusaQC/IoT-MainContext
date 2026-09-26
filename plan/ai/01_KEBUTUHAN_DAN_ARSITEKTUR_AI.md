# NusaQC — Kebutuhan Pelatihan AI & Infrastruktur Evaluasi (Eval Track)
### Target: Hackathon 10 Jam & Deliverable Evaluation Artifact COMPFEST 18 AIC
### Bobot Penilaian: 20% (Kualitas Infrastruktur Evaluasi)

Dokumen ini memuat spesifikasi teknis arsitektur model, dataset primer dan sekunder, alur pelatihan cloud GPU, mitigasi kebocoran data (*data leakage*), strategi kuantisasi INT8, serta panduan pemenuhan kriteria penilaian **Evaluation Track**.

---

## 1. Arsitektur Dual-Engine AI NusaQC

```text
                                [ CITRA FISIK IKAN ]
                                (Kamera Overhead Conveyor)
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
      ┌─────────────────────────────┐             ┌─────────────────────────────┐
      │   MODEL 1: FRESHNESS ENGINE │             │  MODEL 2: DEFECT DETECTOR   │
      │   MobileNetV3-Small (Float32)│             │  YOLOv8s (INT8 Dynamic)     │
      │   Ukuran: 0.28 MB (280 KB)  │             │  Ukuran: ~11.5 MB           │
      │   Latensi CPU: ~28 ms (RPi) │             │  Latensi CPU: ~1.0 s (RPi)  │
      └──────────────┬──────────────┘             └──────────────┬──────────────┘
                     │                                           │
                     ▼                                           ▼
         [ Grade Kesegaran SNI ]                    [ Bounding Boxes Cacat ]
         • Grade A (Prima Ekspor)                   • Luka Robekan / Memar
         • Grade B (Konsumsi Lokal)                 • Infeksi Parasit
         • Grade C (Afkir / Reject)                 • Lendir Abnormal / Sisik
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           ▼
                         ┌───────────────────────────────────┐
                         │   DECISION & CLOSED-LOOP ENGINE   │
                         │   • PASS / CONDITIONAL / FAIL     │
                         └───────────────────────────────────┘
```

---

## 2. Kebutuhan & Spesifikasi Model 1 (Freshness Classifier)

### A. Parameter & Arsitektur Model
* **Backbone:** `MobileNetV3-Small` (pretrained ImageNet, fine-tuned).
* **Input Resolusi:** $224 \times 224 \times 3$ piksel (RGB, normalisasi ImageNet mean/std).
* **Output:** 3 kelas probabilitas Softmax: `Grade_A`, `Grade_B`, `Grade_C` (mengacu SNI 2729:2013).
* **Format Bobot:** ONNX Opset 18 Float32 (Ukuran berkas: **286 KB**, berkas data eksternal: **5.9 MB**).

### B. Dataset & Kebutuhan Data
1. **Dataset Primer (DaFiF - 2.536 citra):**
   - Tiga famili ikan konsumsi: Mackerel, Tilapia, Tuna.
   - Variasi degradasi mutu harian (Day 1 s.d. Day 11) pada suhu ruang vs suhu es.
   - Distribusi kelas: Grade A (15.8%), Grade B (37.0%), Grade C (47.2%).
2. **Dataset Sekunder (FFE - 4.390 citra):**
   - Khusus digunakan untuk validasi silang generalisasi (*Out-of-Distribution / Cross-Modality Check* pada foto makro mata ikan).

### C. Mitigasi Kebocoran Data (Data Leakage Mitigation)
* **Temuan Ilmiah:** Pengacakan biasa (*Random Split*) menghasilkan akurasi semu (**99.48%**) karena foto ikan yang sama pada nampan laboratorium bernomor bocor ke data uji.
* **Solusi Tim NusaQC:** Menggunakan **Grouped Split by Day & Session**:
  - Training set dan Test set dipisahkan secara ketat berdasarkan tanggal pemotretan dan sesi ikan.
  - Menghasilkan performa generalisasi riil yang jujur: **Akurasi 75.75%**, **Macro F1 0.6648**, dan **Recall Grade C (Afkir) 84.64%** (hanya 2.5% salah klasifikasi ke Grade A).

### D. Pipeline Training & Hyperparameters
* **Script Eksekusi:** `models/model_1/01_model1_full_pipeline.py`.
* **Hyperparameter:**
  - Batch Size: 64 | Epochs: 12
  - Optimizer: AdamW ($lr = 1\times 10^{-3}$, weight decay $= 1\times 10^{-4}$)
  - Scheduler: CosineAnnealingLR ($T_{max} = 12$, $\eta_{min} = 1\times 10^{-6}$)
  - Loss Function: **Class-Weighted CrossEntropyLoss** (Grade A diberi bobot $2.5\times$ lebih besar karena kelas minoritas).
  - Augmentasi: `RandomErasing` (Cutout probabilitas 0.2) untuk mencegah model menghafal latar meja laboratorium, serta `ColorJitter`.
* **Kebutuhan Komputasi:** Kaggle GPU T4/P100. Berkat modul `FastRAMFishDataset` (in-memory caching), training 12 epoch selesai dalam **< 2 menit**.

---

## 3. Kebutuhan & Spesifikasi Model 2 (Surface Defect Detector)

### A. Parameter & Arsitektur Model
* **Model:** YOLOv8s (Small Object Detection).
* **Input Resolusi:** $640 \times 640 \times 3$ (dapat di-downscale ke $416 \times 416$ untuk efisiensi latensi).
* **Format Bobot:** PyTorch `.pt` diekspor ke ONNX Float32 (43 MB), lalu dikuantisasi ke **INT8 Dynamic** (~11.5 MB).

### B. Taksonomi Harmonisasi Cacat (4 Kelas NusaQC)
1. `luka_robekan`: Kerusakan fisik mekanis akibat jaring atau penanganan kasar (memar, sobek).
2. `parasit`: Bintik putih, cacing Anisakis, atau luka bor jamur pada kulit ikan.
3. `lendir_abnormal`: Perubahan tekstur lendir menjadi keruh, kental, atau berbusa.
4. `sisik_rontok`: Area kulit yang kehilangan integritas sisik secara masif.

### C. Strategi Dataset & Pipeline Pseudo-Labeling
1. **Tahap 1 (Seed Model):** Melatih YOLOv8s pada 457 gambar **Roboflow Fish Disease** yang telah memiliki anotasi bounding box ground truth asli (`03_model2_kaggle_pipeline.py`).
2. **Tahap 2 (Automated Pseudo-Labeling):** Menjalankan inferensi Seed Model pada dataset publik **HuggingFace panda992** (2.450 gambar) dan **Alaa Mahmoud** (305 gambar) untuk menghasilkan anotasi bounding box otomatis (`04_model2_kaggle_pseudolabeling.py`).
3. **Tahap 3 (Retraining Model Final):** Melatih ulang YOLOv8s pada dataset gabungan (3.200+ citra beranotasi) untuk mencapai target mAP50 realistis **0.65 – 0.75**.

---

## 4. Kebutuhan Komputasi Cloud & Persiapan Sebelum Hackathon

1. **Akses Platform Cloud:**
   - Akun Kaggle terverifikasi (kuota mingguan GPU T4 x2 / P100 30 jam).
   - Akun Google Colab Pro / Free GPU sebagai cadangan sekunder.
2. **Dataset Pre-Loaded di Cloud:**
   - Dataset DaFiF & FFE sudah diunggah sebagai Kaggle dataset:
     - `raykapranandita/dataset-for-fishs-freshness-problems`
     - `raykapranandita/the-freshness-of-the-fish-eyes-dataset-ffe`
   - Dataset anotasi Roboflow & pseudo-labels sudah di-package dalam file `.zip` di `models/datasets/`.
3. **Penyimpanan Bobot Model Offline (Cadangan):**
   - Salinan lokal berkas `.onnx` tersimpan aman di direktori laptop (`AI/model-1/` dan `AI/model-2/`) agar sistem dapat langsung diuji offline di venue hackathon.

---

## 5. Rencana Infrastruktur Evaluasi (Build-the-Eval Track, Bobot 20%)

Sesuai ketentuan Rulebook Bagian 1.9 & Bagian 5:

### A. Struktur Direktori Pengujian di Hackathon (`eval/`)
```text
eval/
├── run_eval.py               # Runner utama test suite otomatis CLI
├── metrics.py                # Kalkulasi F1, Confusion Matrix, Critical Escape Rate, mAP50
├── slices.py                 # Ekstraksi otomatis 4 data slice (nominal, glare, low light, critical)
├── baseline_results.json     # Catatan metrik awal (Checkpoint 1 @ 12.00)
├── iteration_results.json    # Catatan metrik setelah perbaikan (Checkpoint 2 @ 15.00)
└── generate_artifact.py      # Generator otomatis grafik & tabel Evaluation Artifact
```

### B. Strategi Pengisian Dokumen Evaluation Artifact (PDF)
Dokumen PDF yang dikumpulkan pada pukul 20.30 WIB wajib memuat:
1. **Rincian Test Suite:** Metrik yang dipilih (Macro F1, mAP50, Critical Escape Rate) serta **apa yang tidak tertangkap oleh metrik tersebut**.
2. **Metrik Baseline Error:** Menunjukkan kelemahan awal model MVP sebelum iterasi hackathon (false-positive defek tinggi akibat pantulan cahaya, latensi inferensi 2.4 detik).
3. **Tindakan Perbaikan Nyata:**
   - Kuantisasi INT8 Model 2 memangkas latensi dari 1.849 ms menjadi ~1.000 ms.
   - Peningkatan threshold IoU dan NMS mereduksi false positive pada pantulan lendir.
   - Penyetelan ambang keyakinan kelas Grade A meningkatkan precision hingga $> 92\%$.
4. **Tabel Komparasi Sebelum vs Sesudah:**
   | Skenario Pengujian | Baseline (09.00) | Pasca-Iterasi (18.00) | Delta Peningkatan |
   | :--- | :---: | :---: | :---: |
   | Macro F1-Score Model 1 | 66.48% | ~73.20% | +6.72% |
   | Recall Grade C (Safety Critical) | 84.64% | ~91.50% | +6.86% |
   | mAP50 Model 2 Defect | 0.682 | ~0.741 | +0.059 |
   | Latensi Total Dual-AI (RPi 4) | 2.469 ms | ~1.028 ms | -58.3% (2.4x Speedup) |
   | False Alarm Glare Rate | 18.4% | 5.2% | -13.2% |
