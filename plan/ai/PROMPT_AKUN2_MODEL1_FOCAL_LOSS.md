# NusaQC — Master Prompt Akun 2 Sesi 3: Model 1 Freshness (Focal Loss Challenger)
### Target: Menguji Multiclass Focal Loss ($\gamma=2.0$) untuk Mengatasi Ambiguitas Visual Hari ke-4 (Day 4)
### Lingkungan: Kaggle Notebook GPU (T4 / P100) · Alokasi: 09.00 – 14.30 WIB · Peran: Studi Komparasi A/B & Kegagalan Eksperimen

Dokumen ini memuat **Master Prompt Siap Pakai Sekali Copy-Paste (One-Shot)** untuk Akun 2 Sesi 3.

> ⚖️ **Kepatuhan Rulebook Babak Final AIC:**
> - Berstatus sebagai berkas perencanaan kerja (*Hackathon Planning Document* - Bagian 2.1 Poin 3).
> - Seluruh script pelatihan dijalankan **SETELAH pukul 09.00 WIB**.
> - Eksperimen ini dirancang secara sengaja untuk menghasilkan data komparasi saintifik (*Ablation Study & Failed Experiment Analysis*) yang diwajibkan panitia di dokumen *Evaluation Artifact* (Bobot 20%).

---

## 🚀 ONE-SHOT MASTER PROMPT (Copy Seluruh Blok Ini ke AI Assistant)

```text
Bertindaklah sebagai Senior Deep Learning Researcher (Spesialisasi Imbalanced Learning & Loss Engineering) untuk sistem "NusaQC" di babak Final Hackathon COMPFEST 18 AIC.

KONTEKS DAN TUJUAN RISET:
Kita sedang menguji hipotesis kompetitif (Challenger Experiment B1) untuk MODEL 1: FRESHNESS CLASSIFIER (MobileNetV3-Small) pada dataset DaFiF (2.536 citra) dengan 3 kelas SNI: Grade A, Grade B, Grade C.

HIPOTESIS RISET B1:
Pada dataset DaFiF, terdapat ambiguitas visual yang sangat tinggi pada ikan hari ke-4 (Day 4) di mana perubahan fisik mata dan insang berada persis di ambang batas antara Grade A dan Grade B. Cross-Entropy loss standar memperlakukan seluruh sampel secara merata, menyebabkan model bias ke kelas mudah.
Kita menguji hipotesis: "Penerapan Multiclass Focal Loss (gamma=2.0) dengan Dynamic Focusing Parameter akan menurunkan loss pada sampel mudah (easy examples) dan memusatkan gradien bobot pada sampel batas yang ambigu (hard examples pada Day 4), menghasilkan Macro F1 yang lebih tinggi dibanding Weighted Cross-Entropy."

TUGAS ANDA:
Tuliskan satu skrip Python lengkap, modular, dan siap dieksekusi di Kaggle GPU:

FASE 1: DATA AUDIT & IDENTIFIKASI HARD SAMPLES
- Load dataset DaFiF dan lakukan filter metadata untuk mengisolasi subset citra `Day 3`, `Day 4`, dan `Day 5` (fase transisi mutu).
- Terapkan Grouped Stratified Split (berdasarkan Spesies + Sesi Hari) untuk menjamin validasi bebas kebocoran data.
- Analisis distribusi probabilitas model baseline pada subset transisi ini.

FASE 2: MULTICLASS FOCAL LOSS IMPLEMENTATION (PyTorch)
- Buat kelas `FocalLoss(nn.Module)` kustom:
  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
  * Parameter: `gamma = 2.0`, `alpha = torch.tensor([0.25, 0.35, 1.0]).cuda()` (memberi bobot lebih tinggi pada Grade C).
  * Pastikan implementasi numerik stabil menggunakan `log_softmax` dan `clamp` probabilitas untuk mencegah NaN.
- Model: `timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=3)`.
- Optimizer: AdamW(lr=3e-4, weight_decay=1e-4), Scheduler: CosineAnnealingLR(epochs=15), Batch size: 32.
- Latih model selama 15 epoch dan simpan bobot terbaik: `model1_akun2_focal_best.pth`.

FASE 3: A/B TESTING HEAD-TO-HEAD (FOCAL VS WEIGHTED LOSS)
- Uji model Focal Loss ini pada:
  1. Seluruh data validasi DaFiF.
  2. Subset khusus ikan Day 4 (sampel batas).
  3. Irisan `safety_critical` (sampel Grade C).
- Hitung: Macro F1, Akurasi subset Day 4, Recall Grade C, dan Critical Escape Rate (C -> A).
- Bandingkan dengan metrik dari Eksperimen A1 (Weighted Loss):
  * Apakah Focal Loss berhasil mengalahkan Weighted Loss dalam Macro F1?
  * Apakah Focal Loss mampu menekan Critical Escape Rate ke 0.00%?
- Simpan data komparasi ke `eval/ab_testing_model1_results.json`.

FASE 4: ANALISIS ILMIAH KEGAGALAN / KESUKSESAN HIPOTESIS
- Buatkan 2 paragraf evaluasi akademik berbobot saintifik (Bahasa Inggris):
  * Jelaskan alasan kausal mengapa Focal Loss unggul atau justru kalah dibanding Weighted Cross-Entropy (misal: Focal Loss meredam gradien sampel Grade C yang sebenarnya sudah berbobot sedikit sehingga pembaruan bobot untuk kelas kritis menjadi kurang agresif).
  * Format teks ini agar siap dimasukkan ke Bab "Failed Experiments & Ablation Studies" pada PDF Evaluation Artifact.
- Export bobot ke `mobilenetv3_freshness_focal.onnx` untuk keperluan arsip verifikasi juri.
```

---

## 📌 Kode Acuan Multiclass Focal Loss (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Tensor bobot kelas [alpha_0, alpha_1, alpha_2]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```
