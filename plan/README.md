# NusaQC — Master Plan Babak Final Hackathon & Integrasi Sistem
### COMPFEST 18 AI Innovation Challenge (AIC) · Smart Manufacturing Track
### Venue: Gedung Baru Fasilkom UI · Waktu: Sabtu–Minggu, 26–27 September 2026

Seluruh rencana strategis, operasional, arsitektur, dan master prompt telah diaudit dan dipisahkan secara terstruktur ke dalam dua lintasan utama: **AI Track** (`plan/ai/`) dan **IoT Track** (`plan/iot/`).

---

## 🗂️ Peta Navigasi Folder `plan/`

```text
plan/
├── README.md                           # Master Portal (Dokumen ini)
│
├── ai/                                 # 🧠 LINTASAN AI, EVALUASI, & KOMPUTASI CLOUD
│   ├── README.md                       # Indeks AI Track, Ringkasan Metrik, & Setup
│   ├── 01_KEBUTUHAN_DAN_ARSITEKTUR_AI.md # Arsitektur Dual-Model, Dataset DaFiF/Roboflow, Anti-Leakage
│   ├── 02_STRATEGI_PARALEL_2_KAGGLE_4_SESI.md # Blueprint 4 Sesi GPU Paralel (A/B Testing Champion vs Challenger)
│   ├── 03_TEST_SUITE_DAN_EVAL_TRACK.md # Metodologi Slices (Nominal, Glare, Dim, Critical), Kode eval/, Master Prompt
│   ├── PROMPT_AKUN1_MODEL1_WEIGHTED_LOSS.md # 📄 PROMPT 1: Akun 1 Sesi 1 (Model 1 Weighted Loss Zero Escape)
│   ├── PROMPT_AKUN1_MODEL2_GLARE_AUG.md     # 📄 PROMPT 2: Akun 1 Sesi 2 (Model 2 Specular Glare Augmentation)
│   ├── PROMPT_AKUN2_MODEL1_FOCAL_LOSS.md    # 📄 PROMPT 3: Akun 2 Sesi 3 (Model 1 Focal Loss Challenger)
│   └── PROMPT_AKUN2_MODEL2_INT8_LATENCY.md  # 📄 PROMPT 4: Akun 2 Sesi 4 (Model 2 Dynamic INT8 & Kompresi)
│
└── iot/                                # ⚡ LINTASAN IOT, HARDWARE, & EDGE DEPLOYMENT
    ├── README.md                       # Indeks IoT Track & Ringkasan Parameter Cepat
    ├── 01_STRATEGI_OPERASIONAL_HACKATHON.md # SOP Luring, Verifikasi LO Pra-Final, Git Tag Checkpoint
    └── 02_IMPLEMENTASI_HARDWARE_DAN_EDGE.md # Arsitektur Distributed Edge, BOM, Wiring 3.3V, Kalibrasi Sensor
```

---

## 📊 Matriks Target Perbaikan AI & Latensi

| Aspek / Metrik | Baseline Awal | Target Akhir | Akun Penanggung Jawab | Dokumen Detail |
| :--- | :---: | :---: | :---: | :--- |
| **Model 1 Macro F1** | 66.48% | **$\ge 76.50\%$** | Akun 1 Sesi 1 vs Akun 2 Sesi 3 | `ai/PROMPT_AKUN1_MODEL1_WEIGHTED_LOSS.md` |
| **Recall Grade C (Safety)** | 79.10% | **$\ge 95.00\%$** | Akun 1 Sesi 1 | `ai/PROMPT_AKUN1_MODEL1_WEIGHTED_LOSS.md` |
| **Critical Escape Rate (C $\rightarrow$ A)** | 2.54% | **$0.00\%$ (Zero Escape)** | Akun 1 Sesi 1 | `ai/PROMPT_AKUN1_MODEL1_WEIGHTED_LOSS.md` |
| **Model 2 Defect mAP50** | 0.682 | **$\ge 0.735$ (INT8)** | Akun 1 Sesi 2 & Akun 2 Sesi 4 | `ai/PROMPT_AKUN1_MODEL2_GLARE_AUG.md` |
| **False Alarm Glare Rate** | 18.40% | **$\le 5.00\%$** | Akun 1 Sesi 2 | `ai/PROMPT_AKUN1_MODEL2_GLARE_AUG.md` |
| **Latensi Model 1 (RPi 4)** | 28 ms | **~22 - 28 ms** | Akun 1 Sesi 1 | `ai/PROMPT_AKUN1_MODEL1_WEIGHTED_LOSS.md` |
| **Latensi Model 2 (RPi 4)** | 2,440 ms (FP32) | **$\sim 1.000\text{ ms}$ (INT8)** | Akun 2 Sesi 4 | `ai/PROMPT_AKUN2_MODEL2_INT8_LATENCY.md` |
| **Total Latensi Dual-AI RPi 4** | 2,469 ms (FP32) | **$\sim 1.028\text{ ms}$ (INT8)** | Dual Pipeline Lengkap | `ai/02_STRATEGI_PARALEL_2_KAGGLE_4_SESI.md` |
| **Ukuran Model YOLOv8 ONNX**| 43.0 MB | **$\sim 11.5\text{ MB}$ (-73%)** | Akun 2 Sesi 4 | `ai/PROMPT_AKUN2_MODEL2_INT8_LATENCY.md` |
---

## ⏱️ Linimasa Hackathon & Checkpoint Tag Wajib

| Waktu | Kegiatan / Milestone | Git Tag Wajib | Keterangan |
| :---: | :--- | :---: | :--- |
| **08.00 - 08.45 WIB** | Meja Kerja & Verifikasi LO | - | Pemeriksaan working tree clean & RPi clean state |
| **09.00 WIB** | Hackathon Dimulai (Fase I) | - | Setup Kaggle Akun 1 & Akun 2 secara paralel |
| **12.00 WIB** | Checkpoint 1: Baseline | `checkpoint-1-baseline` | Laporan kegagalan awal & kelemahan baseline |
| **12.00 - 15.00 WIB**| Fase II: Iteration | - | Retraining Akun 1 + Kuantisasi Akun 2 |
| **14.30 - 15.00 WIB**| Konvergensi Model | - | Terapkan skrip INT8 ke bobot retrained Akun 1 |
| **15.00 WIB** | Checkpoint 2: Iteration | `checkpoint-2-iteration` | Peningkatan metrik & latensi berhasil terbukti |
| **15.00 - 16.00 WIB**| Sesi Presentasi WIZ.AI | - | Presentasi progres informal (English) |
| **16.00 - 18.00 WIB**| Finalisasi Iterasi | - | Integrasi model final ke Central Laptop & RPi 4 |
| **18.00 WIB** | Checkpoint 3: Integration | `checkpoint-3-integration` | Sistem end-to-end teruji lancar |
| **20.00 WIB** | **CODE FREEZE MUTLAK** | `final-submission` | Batas akhir commit & push |
| **20.30 WIB** | **Evaluation Artifact (PDF)** | - | Batas akhir upload PDF evaluasi ke `compfest.id` |
| **23.59 WIB** | **Pitch Deck (PPTX/PDF)** | - | Batas akhir upload slide ke `compfest.id` |
