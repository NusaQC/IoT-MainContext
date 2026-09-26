---
name: nusaqc-ai-ml-pipeline
description: >
  Kontrak teknis pipeline Machine Learning NusaQC: arsitektur Model 1
  (MobileNetV3-Small freshness classifier) dan Model 2 (YOLOv8s defect
  detector), dataset, anti-leakage split, training hyperparameter,
  ONNX export, kuantisasi INT8, dan metrik evaluasi. Gunakan skill ini
  saat mengerjakan kode di models/, AI/, atau webdev/eval/.
---

## 1. Arsitektur Dual-Engine AI

```
[ CITRA OVERHEAD CONVEYOR ]
          │
  ┌───────┴───────┐
  ▼               ▼
MODEL 1          MODEL 2
MobileNetV3-Small  YOLOv8s (INT8)
224×224 RGB        640×640 letterbox
0.28 MB            ~11.5 MB
~28 ms RPi4        ~1.000 ms RPi4
  │               │
Grade A/B/C      BBoxes cacat
  └───────┬───────┘
          ▼
  Decision Engine
  PASS / CONDITIONAL / FAIL
```

---

## 2. Model 1 — Freshness Classifier (MobileNetV3-Small)

**Script:** `models/model_1/01_model1_full_pipeline.py`

### Arsitektur & I/O
- Backbone: `MobileNetV3-Small` pretrained ImageNet, fine-tuned 3-kelas
- Input: `224×224×3` RGB, normalisasi ImageNet (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`)
- Output: `(3,)` logits → Softmax → `[p_A, p_B, p_C]`
- Format deployment: **ONNX Opset 18 Float32** (`mobilenetv3_freshness.onnx`)
  - Berkas utama: 286 KB; data eksternal: 5.9 MB

### Label Kelas
| Index | Grade | Arti |
|---|---|---|
| 0 | A | Prima — siap ekspor |
| 1 | B | Segar — konsumsi lokal |
| 2 | C | Afkir / Busuk — reject |

### Dataset
| Dataset | Jumlah | Keterangan |
|---|---|---|
| **DaFiF** (primer) | 2.536 citra | Mackerel, Tilapia, Tuna — variasi Day 1–11 |
| **FFE** (sekunder) | 4.390 citra | OOD validation: foto makro mata ikan |

**Kaggle datasets:**
- `raykapranandita/dataset-for-fishs-freshness-problems` (DaFiF)
- `raykapranandita/the-freshness-of-the-fish-eyes-dataset-ffe` (FFE)

### Anti-Leakage Split — KRITIS
> **JANGAN** pakai `train_test_split` random biasa → akurasi semu 99.48% karena foto ikan yang sama di nampan laboratorium bernomor bocor ke test set.

**Solusi wajib: Grouped Split by Day & Session**
```python
# Pisahkan berdasarkan tanggal pemotretan + sesi ikan
# Training set dan Test set TIDAK boleh ada ikan yang sama
groups = df["fish_session_id"]  # unik per ikan per sesi
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
```
Menghasilkan performa jujur: **Acc 75.75%, Macro F1 0.6648, Recall-C 84.64%**

### Hyperparameter Training
```python
batch_size = 64
epochs     = 12
optimizer  = AdamW(lr=1e-3, weight_decay=1e-4)
scheduler  = CosineAnnealingLR(T_max=12, eta_min=1e-6)
loss       = CrossEntropyLoss(weight=class_weights)
# class_weights: Grade A diberi bobot 2.5× (kelas minoritas 15.8%)
```

### Augmentasi
```python
transforms.RandomErasing(p=0.2)   # Cutout — cegah hafal latar lab
transforms.ColorJitter(...)        # Variasi pencahayaan
```

### ONNX Export
```python
torch.onnx.export(
    model, dummy_input,
    "mobilenetv3_freshness.onnx",
    opset_version=18,
    input_names=["x"],
    output_names=["linear_1"],
    dynamic_axes={"x": {0: "batch"}, "linear_1": {0: "batch"}}
)
```

### In-Memory Dataset Cache (Training Kaggle T4 < 2 menit)
```python
class FastRAMFishDataset(Dataset):
    def __init__(self, paths, labels, transform):
        # Load semua PIL image ke RAM saat __init__
        self.images = [Image.open(p).convert("RGB") for p in paths]
        self.labels = labels
        self.transform = transform
```

---

## 3. Model 2 — Surface Defect Detector (YOLOv8s)

**Scripts:**
- `models/model_2/03_model2_kaggle_pipeline.py` — training seed model
- `models/model_2/04_model2_kaggle_pseudolabeling.py` — pseudo-labeling
- `models/model_2/01_prepare_model2_dataset.py` — folder init & label harmonisasi

### Arsitektur & I/O
- Base: **YOLOv8s** (Ultralytics)
- Input: `640×640×3` letterbox-padded; bisa di-downscale ke `416×416` untuk efisiensi
- Output: `(1, channels, 8400)` → transpose → NMS via `cv2.dnn.NMSBoxes` → unletterbox coords
- Format deployment: **ONNX INT8 Dynamic** (`nusaqc_model2_defect_detector.onnx`, ~11.5 MB)

### Taksonomi 4 Kelas Cacat
| Index | Label | Deskripsi |
|---|---|---|
| 0 | `sisik_sisa` | Kehilangan integritas sisik masif (Scale loss, parasit Argulus) |
| 1 | `warna_abnormal` | Bacterial Red Disease, Aeromoniasis, Hemorrhage |
| 2 | `luka_robekan` | Skin ulcer, Fin rot, Saprolegniasis |
| 3 | `lendir_berlebih` | White tail disease, lendir keruh/berbusa |

> ⚠️ **Mapping label di `inference.py` BERBEDA dari dokumen lama** (`plan/ai/`) yang masih menyebut `parasit`/`sisik_rontok`. Ground truth di kode adalah 4 label di atas.

### Pipeline Dataset (3 Tahap)
```
Tahap 1: Seed Model
  → 457 gambar Roboflow Fish Disease (anotasi ground truth asli)
  → Training YOLOv8s → Seed checkpoint

Tahap 2: Pseudo-Labeling Otomatis
  → Inferensi Seed Model pada:
    - HuggingFace panda992: 2.450 gambar
    - Alaa Mahmoud dataset: 305 gambar
  → Menghasilkan 3.509 bounding box pseudo-label

Tahap 3: Retraining Final
  → Dataset gabungan 3.212+ citra beranotasi
  → Target mAP50: 0.65–0.75
```

### ONNX Export & Kuantisasi INT8
```python
# Export dari Ultralytics
model.export(format="onnx", opset=18, dynamic=True)

# Kuantisasi INT8 Dynamic (onnxruntime tools)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    "yolov8s_float32.onnx",
    "nusaqc_model2_defect_detector.onnx",
    weight_type=QuantType.QInt8
)
# Hasil: 43 MB → ~11.5 MB (-73%), latensi RPi4 1.849 ms → ~950 ms (2.2× speedup)
```

### Thread Tuning (RPi4 Cortex-A72)
```python
# Set via env var atau InferenceSession options
# 4 thread > 2 thread untuk Model 2 (-25% latensi)
sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = int(os.getenv("NUSAQC_ORT_THREADS", "4"))
```

---

## 4. Metrik Evaluasi — Target Hackathon

| Metrik | Baseline | Target Final | Lokasi Script |
|---|---|---|---|
| **Macro F1 (Model 1)** | 66.48% | ≥ 76.50% | `models/model_1/` |
| **Recall Grade C (Safety)** | 84.64% (79.10% setelah recheck) | ≥ 95.00% | idem |
| **Critical Escape Rate (C→A)** | 2.54% | **0.00% (Zero Escape)** | idem |
| **mAP50 (Model 2)** | 0.682 | ≥ 0.735 (INT8) | `models/model_2/` |
| **False Alarm Glare Rate** | 18.40% | ≤ 5.00% | idem |
| **Latensi Dual-AI RPi4** | 2.469 ms (FP32) | ~1.028 ms (INT8) | `webdev/eval/` |

### Definisi Critical Escape Rate
```
CER = FN(C→A) / Total_Grade_C_Samples × 100%
```
CER **HARUS 0%** — ikan afkir tidak boleh lolos ke Grade A (risiko keamanan pangan ekspor).

---

## 5. Struktur File Bobot & Path Convention

```
webdev/backend/models_weights/
├── mobilenetv3_freshness.onnx        # Model 1 (float32, < 10 MB combined)
└── nusaqc_model2_defect_detector.onnx # Model 2 (INT8, ~11.5 MB)

AI/
├── model-1/                           # Backup lokal offline (venue hackathon)
└── model-2/

models/datasets/                       # Git ignored — dataset lokal
├── model-1/                           # DaFiF & FFE
└── model-2/                           # Roboflow + pseudo-labels
```

`settings.MODEL_DIR` (di `config.py`) → path ke `models_weights/`. Jika `.onnx` tidak ada → **Simulation Mode** (dummy output, tidak error).

---

## 6. Gotcha & Risiko

| # | Masalah | Mitigasi |
|---|---|---|
| 1 | Random split → leakage 99.48% | SELALU `GroupShuffleSplit` by session |
| 2 | Glare/specular reflection → false positive defek | Augmentasi `RandomBrightness` + NMS IoU ketat (0.45) |
| 3 | Grade C escape ke Grade A | Loss weighted + Recall-C priority metric |
| 4 | Model 2 latensi 2.4 detik FP32 di RPi | WAJIB INT8 + 4 thread |
| 5 | Label mismatch antara dokumen lama vs kode | Gunakan kode sebagai ground truth |
