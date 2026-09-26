# NusaQC — Master Prompt Akun 2 Sesi 4: Model 2 Defect (Kuantisasi INT8 & Latensi)
### Target: Pemangkasan Latensi CPU RPi 4 ke ~1.000 ms (-58%) & Kompresi Model dari 43 MB ke ~11.5 MB
### Lingkungan: Kaggle Notebook CPU / GPU · Alokasi: 09.00 – 14.45 WIB · Output: `nusaqc_model2_defect_detector.onnx`

Dokumen ini memuat **Master Prompt Siap Pakai Sekali Copy-Paste (One-Shot)** untuk Akun 2 Sesi 4.

> ⚖️ **Kepatuhan Rulebook Babak Final AIC:**
> - Berstatus sebagai berkas perencanaan kerja (*Hackathon Planning Document* - Bagian 2.1 Poin 3).
> - Seluruh benchmarking, kuantisasi model, dan integrasi bobot baru dieksekusi **SETELAH pukul 09.00 WIB**.
> - Menghasilkan model deteksi cacat final siap pakai di Raspberry Pi 4 dengan performa sub-detik.

---

## 🚀 ONE-SHOT MASTER PROMPT (Copy Seluruh Blok Ini ke AI Assistant)

```text
Bertindaklah sebagai Senior Edge AI Optimization Engineer (Spesialisasi ARM Cortex-A72, ONNX Runtime, & Model Quantization) untuk proyek "NusaQC" di babak Final COMPFEST 18 AIC.

KONTEKS DAN TUJUAN:
Sesi ini bertanggung jawab atas ENGINE KOMPRESI & LATENSI MODEL 2 (YOLOv8s Defect Detector).
Pada hardware target Raspberry Pi 4 Model B (Quad-Core 1.5 GHz ARM Cortex-A72, 4GB RAM), model YOLOv8s FP32 baseline memiliki:
- Ukuran berkas on disk: 43.0 MB.
- Latensi inferensi CPU: 2.440 ms (hampir 2.5 detik per frame), menyebabkan antrean penumpukan ikan di conveyor.
Kita harus memangkas latensi ini menjadi ~1.000 ms tanpa mengorbankan akurasi pendeteksian cacat (retensi mAP50 >= 98%).

HIPOTESIS PERBAIKAN:
Menerapkan Dynamic Quantization Integer 8-bit (INT8) menggunakan kernel `onnxruntime.quantization` serta konfigurasi multi-threading ARM `intra_op_num_threads=4` akan memangkas memori footprint sebesar ~73% (menjadi ~11.5 MB) dan mempercepat eksekusi forward pass sebesar 2.4x speedup, sekaligus mengevaluasi trade-off jika resolusi diturunkan ke 416x416.

TUGAS ANDA:
Tuliskan satu skrip Python lengkap, modular, dan siap dieksekusi di Kaggle Notebook:

FASE 1: LATENCY PROFILING & INPUT RESOLUTION ABLATION
- Ukur profil latensi baseline YOLOv8s FP32 pada CPU menggunakan 4 thread (`intra_op_num_threads=4`):
  * Pre-processing time (Letterbox resize & normalisasi).
  * Model forward pass time.
  * Post-processing time (NMS & thresholding).
- Lakukan studi ablasi komparasi antara:
  * Opsi A: Input 640x640 (Resolusi standar).
  * Opsi B: Input 416x416 (Resolusi terkompresi).
- Hitung mAP50 pada kelas parasit kecil (< 20 px) vs luka robekan besar. Buat analisis apakah resolusi 416 merusak deteksi parasit. Simpan log ke `eval/resolution_ablation.json`.

FASE 2: DYNAMIC INT8 QUANTIZATION PIPELINE (ONNX RUNTIME)
- Buat fungsi kuantisasi dinamis:
  ```python
  from onnxruntime.quantization import quantize_dynamic, QuantType

  quantize_dynamic(
      model_input="model2_input.onnx",
      model_output="model2_int8.onnx",
      weight_type=QuantType.QInt8,
      op_types_to_quantize=['Conv', 'MatMul', 'Gemm']
  )
  ```
- Hindari kuantisasi node non-linear akhir (seperti Sigmoid/Softmax pada detection head) agar tidak terjadi pergeseran koordinat bounding box.
- Validasi numerik: Hitung Mean Absolute Error (MAE) koordinat bounding box antara FP32 dan INT8 pada 10 citra uji (pastikan selisih MAE < 1.5 piksel).

FASE 3: STATIC CALIBRATION DATA READER (OPSIONAL / QDQ)
- Bangun kelas `FishCalibrationDataReader` untuk membaca 50 citra representatif ikan basah.
- Siapkan fungsi `quantize_static` dengan kalibrasi MinMax untuk pengujian komparasi terhadap dynamic quantization.

FASE 4: KONVERGENSI PUKUL 14.30 WIB (APLIKASIKAN KE BOBOT SESI 2)
- Ambil bobot `model2_akun1_glare_best.onnx` yang dihasilkan oleh Akun 1 Sesi 2 (model dengan mAP tinggi dan bebas false alarm glare).
- Jalankan pipeline kuantisasi INT8 dari Fase 2 terhadap bobot baru tersebut.
- Hasilkan berkas model akhir: `nusaqc_model2_defect_detector.onnx` (format INT8).
- Verifikasi seluruh kriteria penerimaan produk final:
  1. Ukuran berkas on disk: ~11.5 MB (-73% dari 43 MB).
  2. Latensi Forward Pass CPU RPi 4: ~1.000 ms (2.4x speedup).
  3. Retensi mAP50: >= 0.735.
  4. False Alarm Glare Rate: <= 5.0%.
- Ekspor ringkasan akhir ke `eval/model2_int8_final_benchmark.json`.
- Berikan instruksi download berkas untuk dipindahkan ke folder repositori lokal:
  `AI/model-2/nusaqc_model2_defect_detector.onnx`.
```

---

## 📌 Kode Acuan Kuantisasi INT8 & Threading ARM (ONNX Runtime)

```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def create_arm_optimized_session(model_path: str) -> ort.InferenceSession:
    """Mengonfigurasi session options optimal untuk CPU Quad-Core Cortex-A72."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # Menyesuaikan 4 core fisik RPi 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
```
