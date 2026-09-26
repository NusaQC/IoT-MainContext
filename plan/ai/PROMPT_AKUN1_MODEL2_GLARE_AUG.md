# NusaQC — Master Prompt Akun 1 Sesi 2: Model 2 Defect (Specular Glare Augmentation)
### Target: Reduksi False Alarm Pantulan Kilau Air/Lendir dari 18.4% ke <= 5.0% & Kenaikan mAP50 >= 0.735
### Lingkungan: Kaggle Notebook GPU (T4) · Alokasi: 09.00 – 14.30 WIB · Output: `model2_akun1_glare_best.onnx`

Dokumen ini memuat **Master Prompt Siap Pakai Sekali Copy-Paste (One-Shot)** untuk Akun 1 Sesi 2.

> ⚖️ **Kepatuhan Rulebook Babak Final AIC:**
> - Berstatus sebagai berkas perencanaan kerja (*Hackathon Planning Document* - Bagian 2.1 Poin 3).
> - Seluruh pelatihan model dan pembuatan bobot baru dijalankan **SETELAH pukul 09.00 WIB**.
> - Menghasilkan data kuantitatif komparasi sebelum vs sesudah untuk bab "Build-the-Eval Track".

---

## 🚀 ONE-SHOT MASTER PROMPT (Copy Seluruh Blok Ini ke AI Assistant)

```text
Bertindaklah sebagai Senior Object Detection & Computer Vision Engineer untuk sistem inspeksi mutu ikan "NusaQC" di babak Final Hackathon COMPFEST 18 AIC.

KONTEKS DAN TUJUAN SISTEM:
Sesi ini menangani MODEL 2: DEFECT DETECTOR (YOLOv8s) yang bertugas mendeteksi dan melokalisasi 4 kelas cacat fisik pada tubuh ikan di atas conveyor:
- Kelas 0: luka_robekan (Kerusakan mekanik / gigitan predator - Critical Defect)
- Kelas 1: parasit (Cacing anisakis / bintik kontaminasi - Critical Defect)
- Kelas 2: lendir_abnormal (Akumulasi lendir kusam / bercak jamur)
- Kelas 3: sisik_rontok (Kerusakan penanganan pascapanen)

MASALAH PADA BASELINE AWAL (HASIL PENYISIHAN):
1. Overall mAP50 baseline masih berada di angka 0.682.
2. Di lingkungan basah conveyor pabrik, ikan segar dilapisi lendir alami dan air es. Cahaya lampu overhead menciptakan pantulan kilau putih tajam (specular highlights).
3. Pada irisan pengujian `specular_glare`, False Alarm Rate mencapai 18.4%. YOLO baseline salah mengklasifikasikan pantulan putih kilau air sebagai `lendir_abnormal` atau `parasit`.

HIPOTESIS PERBAIKAN:
Mengembangkan generator augmentasi sintetis berbasis Computer Vision (OpenCV) yang menyuntikkan pantulan kilau fotorealistis (Specular Glare Mask) pada 35% citra training ikan sehat TANPA menambahkan anotasi bounding box. Model dipaksa belajar bahwa area kilau putih adalah noise latar belakang normal, sehingga memangkas False Alarm Glare ke <= 5.0% dan menaikkan mAP50 ke >= 0.735.

TUGAS ANDA:
Tuliskan satu skrip Python lengkap, modular, dan siap dieksekusi di Kaggle Notebook GPU T4:

FASE 1: DATASET VERIFICATION & BASELINE GLARE PROFILING
- Lokasi dataset: `/kaggle/input/fish-disease-roboflow/` dan pseudo-labeled data.
- Validasi keberadaan folder `train/`, `val/`, `test/` dan berkas `data.yaml`.
- Buat irisan data uji `specular_glare`: Citra dengan rasio piksel kilau (HSV S <= 40 dan V >= 235) > 1.5% dari luas citra.
- Jalankan evaluasi baseline menggunakan model lama (`nusaqc_model2_defect_baseline.onnx`) pada irisan kilau tersebut.
- Hitung False Alarm Glare Rate awal dan simpan ke `eval/baseline_model2_glare.json`.

FASE 2: SYNTHETIC SPECULAR GLARE AUGMENTATION GENERATOR
- Buat kelas atau fungsi augmentasi OpenCV `apply_specular_glare(img)`:
  * Memilih 2 hingga 6 titik acak pada tubuh ikan.
  * Menggambar elips putih dengan rotasi acak dan sumbu minor 10-30 px, sumbu mayor 30-90 px.
  * Warna pantulan: HSV dengan Saturasi S in [5, 30] (pucat) dan Value V in [240, 255] (sangat terang).
  * Terapkan Gaussian Blur (kernel 11x11, sigma=5) pada tepian elips agar membaur realistis dengan tekstur kulit ikan basah.
- Terapkan augmentasi ini secara on-the-fly atau simpan ke folder training tambahan pada 35% sampel ikan tanpa cacat (background samples).

FASE 3: RETRAINING YOLOV8s (ULTRALYTICS)
- Muat pre-trained model: `YOLO('yolov8s.pt')`.
- Training parameters:
  * `data='data.yaml'`
  * `epochs=25`, `batch=16`, `imgsz=640`
  * `mosaic=1.0`, `mixup=0.15`, `hsv_v=0.4`
  * `box=7.5`, `cls=0.7`, `dfl=1.5`
- Evaluasi metrik validasi setiap epoch dan simpan bobot terbaik: `model2_akun1_glare_best.pt`.

FASE 4: MULTI-SLICE EVALUATION & COMPARISON
- Uji model baru pada data validasi dan irisan `specular_glare`:
  * Hitung Overall mAP50 dan mAP50-95.
  * Hitung False Alarm Glare Rate pada irisan kilau (target: <= 5.0%).
  * Pastikan deteksi kelas kritis (`luka_robekan` dan `parasit`) tidak mengalami penurunan recall.
- Cetak tabel perbandingan metrik Sebelum vs Sesudah.
- Simpan 5 sampel citra hasil visualisasi deteksi berdampingan ke `eval/model2_glare_comparison.jpg`.
- Ekspor ringkasan metrik ke `eval/model2_iteration_results.json`.

FASE 5: ONNX EXPORT UNTUK PIPELINE KUANTISASI
- Ekspor model PyTorch terbaik ke format ONNX Float32:
  `model.export(format="onnx", imgsz=640, dynamic=True, opset=13)`
- Simpan file sebagai `model2_akun1_glare_best.onnx`.
- Berikan instruksi bahwa berkas ini siap di-download dan diteruskan ke pipeline Kuantisasi Dinamis INT8 pada Akun 2 Sesi 4 untuk pemangkasan latensi CPU.
```

---

## 📌 Cuplikan Logika Augmentasi Kilau Sintetis (OpenCV)

```python
def inject_synthetic_glare(image_bgr: np.ndarray, num_glare: int = 3) -> np.ndarray:
    """Menyuntikkan pantulan kilau air/lendir realistis pada citra ikan."""
    h, w = image_bgr.shape[:2]
    glare_layer = np.zeros((h, w, 3), dtype=np.uint8)
    
    for _ in range(num_glare):
        center = (np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4))
        axes = (np.random.randint(25, 60), np.random.randint(8, 20))
        angle = np.random.randint(0, 180)
        # Warna putih kilau (HSV pucat terang)
        cv2.ellipse(glare_layer, center, axes, angle, 0, 360, (255, 255, 255), -1)
        
    glare_blurred = cv2.GaussianBlur(glare_layer, (15, 15), 5)
    result = cv2.addWeighted(image_bgr, 1.0, glare_blurred, 0.75, 0)
    return result
```
