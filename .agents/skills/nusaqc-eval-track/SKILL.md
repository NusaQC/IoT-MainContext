---
name: nusaqc-eval-track
description: >
  Infrastruktur evaluasi NusaQC untuk penilaian Evaluation Track COMPFEST 18 AIC
  (bobot 20%). Mencakup definisi test suite, 4 data slice, metrik prioritas,
  format Evaluation Artifact PDF, dan prosedur checkpoint. Gunakan skill ini
  saat mengerjakan kode di webdev/eval/.
---

## 1. Konteks Penilaian

**Evaluation Track** bernilai **20% total skor** di AIC COMPFEST 18.

Deliverable wajib:
- `checkpoint-1-baseline` (git tag @ 12.00 WIB) — metrik awal sebelum iterasi
- `checkpoint-2-iteration` (git tag @ 15.00 WIB) — bukti peningkatan nyata
- `checkpoint-3-integration` (git tag @ 18.00 WIB) — sistem end-to-end teruji
- **`final-submission`** (git tag @ 20.00 WIB) — CODE FREEZE MUTLAK
- **Evaluation Artifact PDF** (upload @ 20.30 WIB) — dokumen evaluasi
- **Pitch Deck PPTX/PDF** (upload @ 23.59 WIB)

---

## 2. Struktur Direktori Eval

```
webdev/eval/
├── run_all.py                  # CLI runner semua suite: python -m eval.run_all
├── config.py                   # Path resolution — STRICT FileNotFoundError
├── test_async_stream.py        # Suite 5: async conveyor queue simulation
├── results/
│   ├── async_stream_results.json
│   ├── baseline_results.json   # Disimpan saat checkpoint-1
│   └── iteration_results.json  # Disimpan saat checkpoint-2
└── (planned per plan/ai/03_TEST_SUITE_DAN_EVAL_TRACK.md):
    ├── metrics.py              # Kalkulasi F1, Confusion Matrix, CER, mAP50
    ├── slices.py               # Ekstraksi 4 data slice otomatis
    └── generate_artifact.py    # Generator grafik & tabel Evaluation Artifact
```

### Menjalankan Suite
```bash
cd webdev
python -m eval.run_all --suite 5   # async stream saja
python -m eval.run_all             # semua suite
```

---

## 3. Metrik Prioritas

### Model 1 (Freshness Classifier)

| Metrik | Definisi | Target |
|---|---|---|
| **Macro F1** | F1 rata-rata semua kelas (tidak terbobot jumlah sampel) | ≥ 76.50% |
| **Recall Grade C** | $\frac{TP_C}{TP_C + FN_C}$ — seberapa banyak ikan afkir tertangkap | ≥ 95.00% |
| **Critical Escape Rate (CER)** | $\frac{FN_{C \to A}}{Total\_Grade\_C} \times 100\%$ | **0.00%** |
| Precision Grade A | Precision kelas ekspor | > 92% |

**CER adalah metrik safety utama** — satu saja ikan C lolos ke Grade A = kegagalan sistem QC ekspor.

### Model 2 (Defect Detector)

| Metrik | Definisi | Target |
|---|---|---|
| **mAP50** | Mean Average Precision @ IoU 0.50, 4 kelas | ≥ 0.735 (INT8) |
| **False Alarm Glare Rate** | % BBox detected yang bukan defek nyata (akibat glare/refleksi) | ≤ 5.00% |
| Precision per-kelas | Prioritas: `sisik_sisa`, `luka_robekan` | — |

---

## 4. 4 Data Slice Evaluasi

Setiap slice diuji terpisah untuk membuktikan robustness model:

| Slice | Deskripsi | Target Utama |
|---|---|---|
| **Nominal** | Kondisi pencahayaan standar, ikan utuh bersih | Baseline akurasi |
| **Glare** | Foto dengan specular reflection/kilau air tinggi | False alarm rate ≤ 5% |
| **Dim / Low Light** | Pencahayaan redup (kondisi shift malam) | Recall tidak turun > 5% |
| **Critical** | Grade C yang hampir mirip Grade B | CER = 0% wajib |

```python
# slices.py — pattern ekstraksi
def get_critical_slice(dataset):
    """Grade C dengan confidence model awal < 0.6 — paling rawan escape."""
    return [(img, label) for img, label, conf in dataset if label == "C" and conf < 0.6]
```

---

## 5. Format Evaluation Artifact PDF

Dokumen wajib memuat (sesuai Rulebook Bagian 1.9 & 5):

### Bagian 1: Test Suite Description
- Metrik yang dipilih dan **mengapa** — serta apa yang TIDAK tertangkap metrik tersebut
- Contoh: "Macro F1 tidak sensitif terhadap CER karena Grade C adalah kelas minoritas"

### Bagian 2: Baseline Error Analysis (@ Checkpoint 1 — 12.00 WIB)
```
Kelemahan baseline yang didokumentasikan:
- False positive defek tinggi akibat glare (18.4%)
- Latensi inferensi 2.4 detik (FP32, RPi4) — tidak memenuhi SLA ≤ 500 ms
- Macro F1 Model 1: 66.48% (target 76.50%)
- Recall Grade C: 79.10% (target 95%)
```

### Bagian 3: Tindakan Perbaikan Nyata (Evidence-Based)
```
Action 1: Weighted Loss (Grade A bobot 2.5×) → Recall C meningkat
Action 2: INT8 Dynamic Quantization YOLOv8s → latensi 1.849ms → ~1.000ms
Action 3: IoU NMS threshold dinaikkan 0.40→0.45 → false alarm glare turun
Action 4: Glare augmentation (RandomBrightness + CenterCrop) pada training data
```

### Bagian 4: Tabel Komparasi Before/After (WAJIB)

| Skenario | Baseline (09.00) | Pasca-Iterasi (18.00) | Delta |
|---|---|---|---|
| Macro F1 Model 1 | 66.48% | ~76.50% | +10.02% |
| Recall Grade C | 79.10% | ~95.00% | +15.90% |
| Critical Escape Rate | 2.54% | **0.00%** | -2.54% |
| mAP50 Model 2 | 0.682 | ~0.735 | +0.053 |
| Latensi Dual-AI RPi4 | 2.469 ms | ~1.028 ms | -58.3% |
| False Alarm Glare | 18.4% | ~5.0% | -13.4% |

### Bagian 5: Limitations & What's Not Covered
- Hanya 3 famili ikan (Mackerel, Tilapia, Tuna) — belum generalisasi ke semua spesies UPI
- Model 2 pseudo-label belum terverifikasi ahli — confidence bounding box mungkin overstated
- Evaluasi hanya pada dataset indoor lab, belum kondisi factory floor nyata

---

## 6. Confusion Matrix & Kalkulasi Manual

```python
from sklearn.metrics import classification_report, confusion_matrix

# Model 1 output
y_true = [...]  # Grade A/B/C ground truth
y_pred = [...]  # Prediksi model

cm = confusion_matrix(y_true, y_pred, labels=["A", "B", "C"])
report = classification_report(y_true, y_pred, labels=["A", "B", "C"])

# Critical Escape Rate
C_total = cm[2].sum()  # Seluruh sampel Grade C
C_predicted_as_A = cm[2, 0]  # Grade C diklasifikasi sebagai A
CER = (C_predicted_as_A / C_total) * 100
print(f"CER: {CER:.2f}%")  # Harus 0.00%
```

---

## 7. Async Stream Eval (Suite 5)

**File:** `webdev/eval/test_async_stream.py`

Mensimulasikan antrian conveyor asinkron:
- **Burst test:** N frame dikirim bersamaan, `max_workers=2`
- **Endurance test:** 3× 15-frame run berturutan

```python
# Pola wajib — SATU asyncio.run() per eksekusi
def run_async_stream_evaluation(save_results=True) -> dict:
    async def _run_all():
        burst = await simulate_async_conveyor_queue(engine, frames, max_workers=2)
        for _ in range(3):
            await simulate_async_conveyor_queue(engine, frames[:15], max_workers=2)
        return build_payload(burst, ...)
    return asyncio.run(_run_all())
```

**Jangan** multiple `asyncio.run()` — crash di pytest-asyncio context.

**ONNX inference = blocking CPU** → wajib offload ke thread pool:
```python
result = await loop.run_in_executor(None, engine.predict_freshness, frame)
```

---

## 8. Git Tag Checkpoint — SOP Wajib

```bash
# Checkpoint 1 (12.00 WIB) — setelah evaluasi baseline selesai
git add eval/results/baseline_results.json
git commit -m "eval: checkpoint-1 baseline metrics"
git tag checkpoint-1-baseline && git push --tags

# Checkpoint 2 (15.00 WIB) — setelah iterasi model + eval ulang
git add eval/results/iteration_results.json
git commit -m "eval: checkpoint-2 iteration metrics"
git tag checkpoint-2-iteration && git push --tags

# Checkpoint 3 (18.00 WIB) — sistem end-to-end teruji RPi + web
git tag checkpoint-3-integration && git push --tags

# CODE FREEZE (20.00 WIB)
git tag final-submission && git push --tags
```

---

## 9. Eval Config — Path Convention

```python
# webdev/eval/config.py — STRICT: FileNotFoundError jika tidak ada
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "backend" / "models_weights"
FRESHNESS_MODEL_PATH = MODEL_DIR / "mobilenetv3_freshness.onnx"
DEFECT_MODEL_PATH    = MODEL_DIR / "nusaqc_model2_defect_detector.onnx"

DATASET_DIR = Path(r"d:\main\Documents\explore\compe\hackhathon\AIC\models\datasets")
```

Jika menjalankan dari luar `webdev/`, pastikan `sys.path` include:
- `webdev/`
- `webdev/backend/`
