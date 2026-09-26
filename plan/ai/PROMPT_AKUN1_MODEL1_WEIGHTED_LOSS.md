# NusaQC — Master Prompt Akun 1 Sesi 1: Model 1 Freshness (Weighted Loss)
### Target: Eliminasi Critical Escape Rate (Grade C -> Grade A) & Peningkatan Macro F1 >= 76.5%
### Lingkungan: Kaggle Notebook GPU (T4) · Alokasi: 09.00 – 14.30 WIB · Output: `mobilenetv3_freshness.onnx`

Dokumen ini memuat **Master Prompt Siap Pakai Sekali Copy-Paste (One-Shot)** serta rincian modular per sel.

> ⚖️ **Kepatuhan Rulebook Babak Final AIC:**
> - Dokumen ini berstatus sebagai berkas perencanaan kerja (*Hackathon Planning Document* - Bagian 2.1 Poin 3).
> - Seluruh kode program, pelatihan model, dan eksekusi skrip baru dieksekusi **SETELAH pukul 09.00 WIB** di sesi Kaggle aktif.
> - Master prompt ini menginstruksikan AI Assistant untuk menghasilkan kode Python murni tanpa ketergantungan tersembunyi dan mencatat log hasil evaluasi secara otomatis untuk bahan Git Tag `checkpoint-1-baseline` dan `checkpoint-2-iteration`.

---

## 🚀 ONE-SHOT MASTER PROMPT (Copy Seluruh Blok Ini ke AI Assistant)

```text
Bertindaklah sebagai Senior Machine Learning & Computer Vision Engineer untuk proyek "NusaQC" di babak Final Hackathon 10 Jam COMPFEST 18 AIC (Smart Manufacturing Track).

KONTEKS DAN TUJUAN SISTEM:
NusaQC adalah sistem inspeksi mutu ikan otomatis terdistribusi. Sesi ini khusus menangani MODEL 1: FRESHNESS CLASSIFIER (MobileNetV3-Small) yang mengklasifikasikan kesegaran mata dan insang ikan ke dalam 3 kelas standar SNI 2729:2013:
- Kelas 0: Grade_A (Ikan Prima / Standar Ekspor)
- Kelas 1: Grade_B (Ikan Konsumsi Lokal / Penurunan Mutu Ringan)
- Kelas 2: Grade_C (Ikan Busuk / Afkir / Ditolak)

MASALAH PADA BASELINE AWAL (HASIL PENYISIHAN):
1. Macro F1 masih rendah di angka 66.48%.
2. Terdapat Spatiotemporal Data Leakage jika menggunakan Random Split (akurasi semu 99.48%).
3. Terjadi "Critical Escape": 2.54% ikan Grade C (busuk) salah dinilai sebagai Grade A. Di industri ekspor, ini memicu penolakan kontainer oleh FDA/EU.
4. Terdapat ambiguitas visual pada ikan Day 4 (perbatasan Grade A dan B).

HIPOTESIS PERBAIKAN:
Menerapkan Grouped Stratified Split berdasarkan (Spesies + Hari/Sesi) dan melatih ulang MobileNetV3-Small menggunakan Cost-Sensitive Weighted Cross-Entropy Loss dengan bobot penalti berat pada Grade C: W = [1.0, 1.2, 5.0], disertai augmentasi fotometrik untuk mengunci Critical Escape Rate ke 0.00% dan mendongkrak Macro F1 >= 76.50%.

TUGAS ANDA:
Tuliskan satu skrip Python lengkap, modular, dan siap dieksekusi dari awal sampai akhir di Kaggle Notebook (GPU T4) tanpa placeholder atau kode terpotong, yang mencakup 5 FASE LENGKAP:

FASE 1: DATA PREPARATION & ANTI-LEAKAGE SPLIT
- Lokasi dataset: `/kaggle/input/dafif-fish-dataset/` atau direktori data lokal DaFiF (2.536 citra) dan FFE.
- Baca metadata spesies, hari (Day 1 - 7), dan sesi.
- Pisahkan data menggunakan Grouped Stratified Split (80% Train, 20% Val) di mana seluruh sampel dari satu sesi hanya berada di Train ATAU Val.
- Bangun evaluasi baseline error terhadap model lama (`mobilenetv3_freshness_baseline.onnx`) dan simpan metrik awalnya ke `eval/baseline_model1.json`.

FASE 2: COST-SENSITIVE TRAINING PIPELINE (PyTorch)
- Model: `timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=3)`.
- Ganti classifier head dengan Linear layer + Dropout(0.3).
- Loss Function: `nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2, 5.0]).cuda())`.
- Augmentasi Train:
  * ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
  * RandomAffine(degrees=15, translate=(0.05, 0.05))
  * RandomErasing(p=0.2, scale=(0.02, 0.15))
  * Resize ke (224, 224) dan Normalisasi ImageNet.
- Hyperparameters: AdamW(lr=3e-4, weight_decay=1e-4), CosineAnnealingWarmRestarts(T_0=5, T_mult=2), Batch size=32, Epochs=15.
- Validasi setiap epoch dan simpan checkpoint bobot terbaik: `model1_freshness_retrained.pth` (pilih epoch dengan Macro F1 tertinggi dan Escape Rate = 0%).

FASE 3: MULTI-SLICE STRESS-TESTING (BUILD-THE-EVAL)
- Partisi data uji menjadi 3 irisan:
  * `nominal`: Rata-rata luminansi grayscale Y antara 80 dan 180.
  * `low_light`: Rata-rata luminansi Y < 65 (underexposed).
  * `safety_critical`: Seluruh sampel ground truth Grade C.
- Hitung metrik per-irisan: Accuracy, Macro F1, Recall Grade C, dan Critical Escape Rate (Grade C diprediksi Grade A).
- Cetak tabel komparasi matematis sebelum vs sesudah (Delta Kenaikan).
- Simpan kurva loss dan Confusion Matrix berdampingan ke `eval/model1_confusion_matrix.png`.
- Ekspor ringkasan hasil ke `eval/model1_iteration_results.json` (untuk bahan Git Tag checkpoint-2-iteration).

FASE 4: ONNX EXPORT & CPU BENCHMARKING
- Ekspor model PyTorch terbaik ke format ONNX:
  * Output file: `mobilenetv3_freshness.onnx`
  * Input tensor: `dummy_input = torch.randn(1, 3, 224, 224, device='cuda')`
  * Opsimasi opset=13, dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
- Verifikasi ONNX checker dan uji kesesuaian numerik antara PyTorch vs ONNX Runtime (pastikan selisih absolut maks < 1e-4).
- Benchmark latensi inferensi CPU: Jalankan 100 iterasi di CPU (target: 20 - 28 ms).
- Konfirmasi ukuran berkas on disk tetap ringkas (~280 KB).

FASE 5: COMMIT MESSAGE & SUMMARY ARTIFACT
- Cetak pesan commit Git resmi untuk terminal lokal: `git commit -m "feat(ai): retrain model1 with cost-sensitive loss, achieve zero critical escape rate"` dan tag: `checkpoint-2-iteration`.
- Buatkan 1 paragraf teks penjelasan saintifik (Bahasa Inggris) yang merangkum kausalitas mengapa Weighted Loss berhasil mengeliminasi escape rate tanpa merusak recall Grade A, siap dimasukkan ke naskah Evaluation Artifact.
```

---

## 📌 Rincian Modular per Sel (Jika Dijalankan Bertahap di Notebook)

### Sel 1: Inisialisasi & Verifikasi Dataset DaFiF
```python
import os, cv2, json, time, torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = Path("/kaggle/input/dafif-fish-dataset")
print(f"[INIT] Checking CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}")
```

### Sel 2: Class-Weighted Loss Definition & Training Loop
```python
import torch.nn as nn
weights = torch.tensor([1.0, 1.2, 5.0], dtype=torch.float32).cuda()
criterion = nn.CrossEntropyLoss(weight=weights)
print(f"[SETUP] Cost-Sensitive Loss initialized with weights: {weights.tolist()}")
```

### Sel 3: Ekspor ONNX & Validasi Numerik
```python
# Ekspor ONNX Float32
torch.onnx.export(
    best_model,
    torch.randn(1, 3, 224, 224).cuda(),
    "mobilenetv3_freshness.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=13
)
print("[EXPORT] Model exported to mobilenetv3_freshness.onnx (Size: ~280 KB)")
```
