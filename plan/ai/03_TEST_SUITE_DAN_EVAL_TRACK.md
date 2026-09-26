# NusaQC — Panduan Teknis Data Slices & Master Prompt Eval Track
### Target: Build-the-Eval Track, Baseline Error Discovery, & Evaluation Artifact (PDF)
### Bobot Penilaian: 20% (Kualitas Infrastruktur Evaluasi) + 30% (Live Pitching & Q&A)

Dokumen ini memuat:
1. **Penjelasan Teknis & Algoritmik:** Bagaimana `eval/slices.py` memperoleh dan memisahkan setiap irisan data (*nominal*, *specular glare*, *low light*, *safety-critical*).
2. **Implementasi Lengkap Script Pengujian:** Arsitektur kode Python siap pakai untuk `eval/slices.py`, `eval/metrics.py`, dan `eval/run_eval.py`.
3. **Koleksi 5 Master Prompt Siap Pakai:** Prompt untuk AI coding assistant selama rentang 10 jam lomba.

---

## 1. Bagaimana `eval/slices.py` Memperoleh Setiap Slice?

Rulebook COMPFEST 18 Bagian 2.7 Poin 1 menyatakan:
> *"Sistem computer vision diuji lewat irisan ketahanan seperti pencahayaan, sudut, dan oklusi. Juri menilai ketepatan metodologi terhadap karakteristik sistem, bukan jumlah kasus uji."*

Sistem inspeksi ikan conveyor memiliki 2 tantangan lingkungan utama: **pantulan kilau air/lendir pada pencahayaan overhead** dan **fluktuasi intensitas lampu pabrik**. Oleh karena itu, data uji di direktori `data/` dipartisi secara objektif menggunakan **kombinasi filter label dan ekstraksi sinyal citra (Computer Vision Feature Extraction)**:

```text
                                [ SELURUH CITRA UJI (data/) ]
                                               │
         ┌─────────────────────────┬───────────┴───────────┬─────────────────────────┐
         ▼                         ▼                       ▼                         ▼
 ┌───────────────┐         ┌───────────────┐       ┌───────────────┐         ┌───────────────┐
 │ SAFETY-CRIT   │         │ LOW-LIGHT     │       │ SPECULAR-GLARE│         │ NOMINAL       │
 │ Label-Driven  │         │ Luminance < 65│       │ Glare Ratio   │         │ Kontrol       │
 │ Grade C /     │         │ (Under-       │       │ > 1.5% Area   │         │ 80 ≤ Y ≤ 180  │
 │ Defek Berat   │         │  exposure)    │       │ (Kilau Lendir)│         │ Glare < 0.5%  │
 └───────────────┘         └───────────────┘       └───────────────┘         └───────────────┘
```

---

### A. Slice 1: `safety_critical` (Ikan Busuk & Cacat Berat)
* **Cara Memperoleh:** **Label-Driven Metadata Query**.
* **Logika:** Mengambil seluruh sampel uji yang memiliki ground-truth:
  - Model 1: Label kelas `Grade C` (ikan busuk/afkir menurut standar SNI 2729:2013).
  - Model 2: Bounding box cacat kelas `luka_robekan` dan `parasit` (kontaminasi fisik penyebab penolakan ekspor FDA).
* **Implementasi Kode:**
  ```python
  def is_safety_critical(row):
      # Model 1 Freshness
      if row.get('grade') == 'Grade C':
          return True
      # Model 2 Defect
      if any(d in ['luka_robekan', 'parasit'] for d in row.get('defects', [])):
          return True
      return False
  ```
* **Urgensi Industri:** Mengukur *Critical Escape Rate* (False Negative Rate pada Grade C). Jika ikan Grade C salah dinilai sebagai Grade A, akibatnya adalah penolakan kontainer ekspor di pelabuhan tujuan (bencana finansial bagi UPI).

---

### B. Slice 2: `low_light` (Pencahayaan Redup / Shadowing)
* **Cara Memperoleh:** **Luminance Signal Measurement (OpenCV)** atau **Deterministic Exposure Scaling**.
* **Logika Fisik:** Di pabrik nyata, conveyor sering kali berada di bawah bayangan tubuh pekerja atau lampu neon yang redup ($< 50$ lux). Mata dan insang ikan tampak lebih gelap dari aslinya, berisiko menurunkan akurasi Model 1.
* **Metode 1: Pemindaian Otomatis Citra Eksisting (Mean Luminance)**
  Konversi citra ke *Grayscale* atau kanal $Y$ pada ruang warna YUV:
  $$Y = 0.299R + 0.587G + 0.114B$$
  $$\bar{Y} = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W Y(i, j)$$
  Jika $\bar{Y} < 65$ (pada rentang 0–255), citra diklasifikasikan ke dalam `slice_low_light`.
* **Metode 2: Synthetic Dimming (Stress-Testing Perturbation)**
  Jika jumlah foto redup di dataset asli terbatas, terapkan penggelapan deterministik saat evaluasi:
  ```python
  def apply_low_light_perturbation(img):
      # Scaling pixel intensity by factor 0.45 (simulasi redup 40-50 lux)
      return np.clip(img.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
  ```

---

### C. Slice 3: `specular_glare` (Pantulan Kilau Air & Lendir Basah)
* **Cara Memperoleh:** **Color Space Thresholding (HSV High-Value & Low-Saturation)**.
* **Logika Fisik:** Ikan segar dilapisi lendir alami (*mucus*) dan air es basah. Cahaya lampu halogen/LED overhead yang memantul langsung ke kamera menciptakan bintik kilau putih tajam (*specular highlights*). YOLO sering keliru menganggap pantulan putih ini sebagai bercak jamur/parasit atau lendir abnormal (*False Alarm*).
* **Algoritma Ekstraksi Glare Mask di OpenCV:**
  1. Konversi citra BGR ke HSV ($H, S, V$).
  2. Definisikan piksel *specular glare*: Nilai kecerahan sangat tinggi ($V \ge 235$) dan saturasi warna sangat rendah/pucat ($S \le 40$).
  3. Hitung rasio piksel kilau terhadap total luas citra:
  $$\text{Glare Ratio} = \frac{\sum (S \le 40 \land V \ge 235)}{H \times W}$$
  4. Jika $\text{Glare Ratio} \ge 1.5\%$, citra dimasukkan ke `slice_specular_glare`.
* **Implementasi Kode:**
  ```python
  def is_specular_glare(img, threshold_ratio=0.015):
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      s = hsv[:, :, 1]
      v = hsv[:, :, 2]
      glare_mask = (s <= 40) & (v >= 235)
      ratio = np.count_nonzero(glare_mask) / (img.shape[0] * img.shape[1])
      return ratio >= threshold_ratio, ratio
  ```

---

### D. Slice 4: `nominal` (Kondisi Kontrol Standar)
* **Cara Memperoleh:** Citra dengan parameter pencahayaan dan pantulan normal:
  $$80 \le \bar{Y} \le 180 \quad \text{dan} \quad \text{Glare Ratio} < 0.5\%$$
* **Fungsi:** Menjadi *baseline anchor*. Kinerja pada slice nominal dibandingkan langsung dengan kinerja pada slice ekstrem (*glare* & *low-light*) untuk mengukur ketahanan (*robustness gap*).

---

## 2. Implementasi Lengkap Kode `eval/slices.py`

Berikut adalah skrip referensi yang siap dijalankan di repositori:

```python
"""
eval/slices.py — Data Slicing Module for NusaQC Industrial Stress-Testing
Mengelompokkan data uji menjadi 4 irisan: nominal, low_light, specular_glare, safety_critical.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def extract_image_signals(img: np.ndarray) -> Tuple[float, float]:
    """Ekstraksi mean luminance dan specular glare ratio dari citra BGR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    glare_mask = (s <= 40) & (v >= 235)
    glare_ratio = float(np.count_nonzero(glare_mask) / (img.shape[0] * img.shape[1]))
    
    return mean_lum, glare_ratio

def partition_slices(data_df: pd.DataFrame, data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Mempartisi DataFrame data uji ke dalam 4 irisan evaluasi."""
    nominal_rows, low_light_rows, glare_rows, critical_rows = [], [], [], []
    
    for idx, row in data_df.iterrows():
        img_path = data_dir / row['image_filename']
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        mean_lum, glare_ratio = extract_image_signals(img)
        
        # 1. Safety Critical (Grade C atau cacat berat)
        if row.get('grade') == 'Grade C' or any(d in ['luka_robekan', 'parasit'] for d in row.get('defects', [])):
            critical_rows.append(row)
            
        # 2. Low Light (Luminance < 65)
        if mean_lum < 65.0:
            low_light_rows.append(row)
            
        # 3. Specular Glare (Glare Ratio > 1.5%)
        if glare_ratio >= 0.015:
            glare_rows.append(row)
            
        # 4. Nominal (Normal Baseline)
        if 80.0 <= mean_lum <= 180.0 and glare_ratio < 0.005:
            nominal_rows.append(row)
            
    return {
        "nominal": pd.DataFrame(nominal_rows),
        "low_light": pd.DataFrame(low_light_rows),
        "specular_glare": pd.DataFrame(glare_rows),
        "safety_critical": pd.DataFrame(critical_rows)
    }
```

---

## 3. Koleksi 5 Master Prompt Siap Pakai

Gunakan kelima prompt ini secara berurutan sesuai timeline hackathon:

---

### 🟢 PROMPT 1: Pembangunan Kerangka Test Suite Otomatis (`eval/`)
**Waktu:** 09.00 – 10.30 WIB (Awal Fase I Baseline)

```text
Bertindaklah sebagai Senior AI QA Engineer untuk sistem "NusaQC" (COMPFEST 18 AIC). 

Konteks Sistem:
- Model 1: Freshness Classifier (MobileNetV3-Small Float32 ONNX, input 224x224, 3 kelas SNI: Grade_A, Grade_B, Grade_C).
- Model 2: Defect Detector (YOLOv8s ONNX, input 640x640/416x416, 4 kelas: luka_robekan, parasit, lendir_abnormal, sisik_rontok).
- Folder data uji: `data/` memuat foto ikan beserta anotasi bounding box dan label kesegaran.

Tugas:
Bangun kerangka pengujian otomatis modular di dalam folder `eval/` yang memenuhi ketentuan "Build-the-Eval Track" (bobot 20%):
1. `eval/slices.py`: Partisi data uji ke 4 irisan: nominal (kontrol), low_light (mean luminance < 65), specular_glare (glare ratio > 1.5%), dan safety_critical (Grade C / defek berat).
2. `eval/metrics.py`: Hitung Macro F1, Per-class Recall/Precision, mAP50, Critical Escape Rate (Grade C diprediksi Grade A), False Positive Glare Rate, dan Latensi CPU (ms).
3. `eval/run_eval.py`: CLI runner yang mengeksekusi inferensi ONNX Runtime, menampilkan tabel metrik rapi di konsol, dan menyimpan ringkasan ke `eval/baseline_results.json`.
4. `eval/generate_artifact.py`: Generator otomatis grafik perbandingan metrik antar-slice dan confusion matrix untuk bahan Evaluation Artifact (PDF).

Pastikan kode berjalan murni di CPU (tanpa GPU dependency) dan robust terhadap missing files.
```

---

### 🟡 PROMPT 2: Analisis Baseline Error & Identifikasi Kelemahan Spesifik
**Waktu:** 11.00 – 12.00 WIB (Persiapan Checkpoint 1 & Sesi WIZ.AI)

```text
Bertindaklah sebagai Lead AI Researcher NusaQC. Kami telah menjalankan eksekusi pertama dari test suite otomatis terhadap model MVP awal kami.

Output `eval/baseline_results.json`:
- Model 1 Macro F1: 66.48% (Grade A: 78.2%, Grade B: 42.1%, Grade C: 79.1%).
- Critical Escape Rate (Grade C -> Grade A): 2.54%.
- Slice Low Light Model 1: Akurasi turun drastis dari 75.7% ke 58.2%.
- Model 2 Overall mAP50: 0.682.
- False Positive Glare Rate pada Model 2: 18.4% (pantulan kilau lendir terdeteksi sebagai lendir_abnormal/parasit).
- Latensi Total Inferensi Dual-AI di RPi 4 (CPU FP32): 2.469 ms (Model 1: 28 ms, Model 2: 2.441 ms).

Tugas:
1. Susun dokumen analisis kelemahan baseline (`eval/BASELINE_ERROR_REPORT.md`):
   - Jelaskan 3 kelemahan terbesar sistem secara tajam berbasis kondisi lapangan di pabrik ikan.
   - Tunjukkan bukti kuantitatif untuk setiap kelemahan.
   - Tetapkan target kuantitatif untuk Fase II Iteration.
2. Buatkan teks elevator pitch 1 paragraf dalam Bahasa Inggris untuk dipaparkan kepada tim WIZ.AI pada jam 15.00 WIB.
3. Buatkan pesan commit Git resmi: `checkpoint-1-baseline`.
```

---

### 🔵 PROMPT 3: Eksekusi Iterasi Perbaikan Model (Evaluation Track)
**Waktu:** 12.00 – 15.00 WIB (Fase II Iteration)

```text
Bertindaklah sebagai Senior CV Optimization Engineer. Kami berada di Fase II Hackathon (Iteration).

Berdasarkan temuan kelemahan Baseline pada Checkpoint 1, implementasikan 3 perbaikan nyata:
1. Kuantisasi Dinamis INT8 Model 2: Konversi YOLOv8s ONNX ke INT8 via `onnxruntime.quantization.quantize_dynamic` (target: memangkas model dari 43 MB ke ~11.5 MB dan latensi CPU RPi 4 ke ~1.000 ms).
2. Perbaikan False Positive Glare Model 2: Training ulang dengan Specular Glare Augmentation dan tuning NMS IoU threshold untuk memangkas false alarm dari 18.4% ke <= 5.0%.
3. Eliminasi Critical Escape Model 1: Retraining MobileNetV3-Small dengan Cost-Sensitive Loss (W_Grade_C = 5.0) sehingga Critical Escape Rate Grade C -> Grade A turun dari 2.54% ke 0.00%.
Tugas:
1. Tuliskan kode perbaikan tersebut dan jalankan ulang `eval/run_eval.py` untuk menyimpan hasil ke `eval/iteration_results.json`.
2. Buatkan tabel komparasi matematis sebelum vs sesudah (Delta Peningkatan).
3. Buatkan pesan commit resmi: `checkpoint-2-iteration`.
```

---

### 🟣 PROMPT 4: Penyusunan Dokumen PDF "Evaluation Artifact"
**Waktu:** 17.30 – 20.00 WIB (Fase III Integration & Pasca Freeze)

```text
Bertindaklah sebagai Lead Technical Author. Kami harus mengunggah deliverable wajib "Evaluation Artifact" dalam format PDF ke portal compfest.id sebelum pukul 20.30 WIB (Bobot 20%).

Rulebook Bagian 4 & 5 mewajibkan dokumen PDF memuat:
1. Rincian Test Suite: Cakupan data slices (nominal, glare, low light, critical), metrik yang diukur, dan alasan pemilihannya (serta batasannya).
2. Temuan Baseline Error: Bukti kegagalan sistem di awal hackathon.
3. Perbandingan Sebelum vs Sesudah: Analisis kausalitas kenaikan metrik (Macro F1 naik +6.72%, recall Grade C naik +6.86%, false glare turun -13.2%, latensi turun 2.4x).
4. Unresolved Failure Modes: Kegagalan yang belum selesai diperbaiki (ambiguitas Grade B pada Day 4) dan rencana mitigasi skala industri.

Tugas:
Tuliskan draf naskah lengkap Evaluation Artifact dalam format Markdown akademik profesional yang siap dirender menjadi PDF.
```

---

### 🔴 PROMPT 5: Simulasi Tanya-Jawab Juri AI Engineer (Q&A Defense)
**Waktu:** 21.00 – 23.00 WIB (Malam Hari H-1 Pitching)

```text
Bertindaklah sebagai Juri AI Engineer yang sangat kritis dan skeptis di ajang AIC COMPFEST 18.

Kaji sistem NusaQC (Dual AI MobileNetV3 + YOLOv8s pada edge Raspberry Pi 4).

Ajukan 5 pertanyaan paling menjebak seputar:
1. Data leakage DaFiF dan keabsahan Grouped Split.
2. Dampak kuantisasi INT8 pada arsitektur Cortex-A72 tanpa instruksi silikon SDOT/UDOT.
3. Alasan memilih Macro F1 dibanding Cost-Sensitive Loss.
4. Domain shift pada spesies ikan di luar dataset DaFiF.
5. Reliabilitas sensor optik inframerah di lingkungan conveyor basah.

Untuk setiap pertanyaan, sertakan:
- Celah teknis yang diincar oleh juri.
- Rekomendasi "Golden Answer" berbasis data empiris yang meyakinkan.
```
